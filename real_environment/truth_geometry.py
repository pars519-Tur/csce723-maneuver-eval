"""Topocentric geometry and Smith-compatible physical FOV gating for external truth."""

from __future__ import annotations

from dataclasses import dataclass
import math

import ephem
import numpy as np


WGS84_SEMI_MAJOR_AXIS_KM = 6378.137
WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3


@dataclass(frozen=True)
class ObserverSite:
    name: str
    longitude_deg: float
    latitude_deg: float
    elevation_m: float

    def pyephem_observer(self, time: ephem.Date) -> ephem.Observer:
        observer = ephem.Observer()
        observer.lon = str(self.longitude_deg)
        observer.lat = str(self.latitude_deg)
        observer.elevation = self.elevation_m
        observer.date = ephem.Date(time)
        return observer


def observer_position_teme_km(site: ObserverSite, time: ephem.Date) -> np.ndarray:
    observer = site.pyephem_observer(time)
    latitude = math.radians(site.latitude_deg)
    local_sidereal_angle = float(observer.sidereal_time())
    altitude_km = site.elevation_m / 1000.0
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_KM / math.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * math.sin(latitude) ** 2
    )
    equatorial_radius = (prime_vertical_radius + altitude_km) * math.cos(latitude)
    return np.array(
        [
            equatorial_radius * math.cos(local_sidereal_angle),
            equatorial_radius * math.sin(local_sidereal_angle),
            (
                prime_vertical_radius * (1.0 - WGS84_ECCENTRICITY_SQUARED)
                + altitude_km
            )
            * math.sin(latitude),
        ]
    )


def topocentric_az_alt_deg(
    target_position_teme_km: np.ndarray,
    site: ObserverSite,
    time: ephem.Date,
) -> tuple[float, float]:
    """Convert a TEME-like target position to PyEphem-style azimuth/elevation."""
    target_position = np.asarray(target_position_teme_km, dtype=float).reshape(3)
    observer = site.pyephem_observer(time)
    sidereal_angle = float(observer.sidereal_time())
    latitude = math.radians(site.latitude_deg)
    line_of_sight = target_position - observer_position_teme_km(site, time)
    line_of_sight /= np.linalg.norm(line_of_sight)

    east = np.array([-math.sin(sidereal_angle), math.cos(sidereal_angle), 0.0])
    north = np.array(
        [
            -math.sin(latitude) * math.cos(sidereal_angle),
            -math.sin(latitude) * math.sin(sidereal_angle),
            math.cos(latitude),
        ]
    )
    up = np.array(
        [
            math.cos(latitude) * math.cos(sidereal_angle),
            math.cos(latitude) * math.sin(sidereal_angle),
            math.sin(latitude),
        ]
    )
    east_component = float(np.dot(line_of_sight, east))
    north_component = float(np.dot(line_of_sight, north))
    up_component = float(np.dot(line_of_sight, up))
    azimuth_deg = math.degrees(math.atan2(east_component, north_component)) % 360.0
    geometric_altitude_deg = math.degrees(
        math.asin(np.clip(up_component, -1.0, 1.0))
    )
    altitude_deg = geometric_altitude_deg + atmospheric_refraction_deg(
        geometric_altitude_deg,
        pressure_mbar=float(observer.pressure),
        temperature_c=float(observer.temp),
    )
    return azimuth_deg, altitude_deg


def atmospheric_refraction_deg(
    geometric_altitude_deg: float,
    pressure_mbar: float = 1010.0,
    temperature_c: float = 15.0,
) -> float:
    """Approximate PyEphem's apparent-altitude correction in the usable sky region."""
    if pressure_mbar <= 0.0 or geometric_altitude_deg <= -1.0:
        return 0.0
    corrected_argument_deg = geometric_altitude_deg + 10.3 / (
        geometric_altitude_deg + 5.11
    )
    refraction_arcminutes = 1.02 / math.tan(math.radians(corrected_argument_deg))
    refraction_arcminutes *= (pressure_mbar / 1010.0) * (
        283.0 / (273.0 + temperature_c)
    )
    return refraction_arcminutes / 60.0


def smith_fov_mask(
    azimuth_deg: np.ndarray,
    altitude_deg: np.ndarray,
    past_pointing_azimuth_deg: float,
    azimuth_action: int,
    altitude_action: int,
) -> np.ndarray:
    """Apply Smith's original 4-degree action-cell geometry to external truth angles."""
    azimuth = np.asarray(azimuth_deg, dtype=float)
    altitude = np.asarray(altitude_deg, dtype=float)
    above_horizon = altitude > 14.0

    relative_azimuth = (
        azimuth + 180.0 - 2.0 - float(past_pointing_azimuth_deg)
    ) % 360.0
    altitude_center = altitude_action * 4.0 + 16.0
    azimuth_center = azimuth_action * 4.0 + 2.0
    valid_altitude = np.abs(altitude - altitude_center) <= 2.0

    cos_altitude_squared = np.cos(np.radians(altitude)) ** 2
    denominator = np.maximum(cos_altitude_squared, np.finfo(float).eps)
    arccos_argument = (
        cos_altitude_squared - 1.0 + math.cos(math.radians(2.0))
    ) / denominator
    effective_azimuth_half_width = np.minimum(
        np.degrees(np.arccos(np.clip(arccos_argument, -1.0, 1.0))), 180.0
    )

    # Intentionally preserve Smith's non-circular absolute-difference comparison.
    valid_azimuth = (
        np.abs(relative_azimuth - azimuth_center) <= effective_azimuth_half_width
    )
    return above_horizon & valid_altitude & valid_azimuth

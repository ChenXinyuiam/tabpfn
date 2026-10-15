"""Strict task contracts for the versioned system workflow."""

from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1
ORIGINAL_ORIGINS = {"measured", "simulated"}


@dataclass(frozen=True)
class TargetSpec:
    column: str
    name: str
    quantity: str
    unit: str
    component: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("Every target field must be a nonempty explicit string")
        if self.quantity not in {"force", "velocity", "unverified"}:
            raise ValueError("quantity must be force, velocity or unverified")
        if self.quantity == "force" and self.unit not in {"N", "mN", "uN", "nN", "pN"}:
            raise ValueError("Force unit must be N, mN, uN, nN or pN")
        if self.quantity == "velocity" and self.unit not in {"m/s", "mm/s", "um/s"}:
            raise ValueError("Velocity unit must be m/s, mm/s or um/s")

    def to_dict(self):
        return asdict(self)

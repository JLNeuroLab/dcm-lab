import numpy as np

from dcm.simulate.design import make_time_grid, boxcar, events, stack_inputs, InputDesign
from dcm.models.neuronal_bilinear import BilinearParameters, BilinearNeuronalModel
from dcm.models.hemodynamic_balloon import HemodynamicParameters, HemodynamicBalloonModel
from dcm.models.forward import ForwardModel

def build_design(cfg: dict) -> InputDesign:
    T = float(cfg["simulation"]["T"])
    dt = float(cfg["simulation"]["dt"])
    t = make_time_grid(T=T, dt=dt)

    names = cfg["inputs"]["names"]
    regs = []
    for name in names:
        spec = cfg["inputs"][name]
        typ = spec["type"]

        if typ == "boxcar":
            regs.append(
                boxcar(
                    t,
                    onsets=spec["onsets"],
                    durations=spec["durations"],
                    amplitudes=spec.get("amplitudes", 1.0),
                )
            )
        
        elif typ == "events":
            # Required: onsets
            # Optional: amplitudes (scalar or list), mode ("nearest" or "floor")
            if "durations" in spec:
                raise ValueError(f"events input '{name}' should not define 'durations'")

            regs.append(
                events(
                    t,
                    onsets=spec["onsets"],
                    amplitudes=spec.get("amplitudes", 1.0),
                    mode=spec.get("mode", "nearest")
                )
            )
        
        else:
            raise ValueError(f"Unknown input type '{typ}' for input '{name}'")

    U = stack_inputs(*regs)  # (T, m)
    return InputDesign(t=t, U=U, names=names)


def build_model(cfg: dict, param_key: str = "neuronal") -> ForwardModel:
    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    # Use the provided key to read parameters
    A = np.asarray(cfg[param_key]["A"], dtype=float)
    B = np.asarray(cfg[param_key]["B"], dtype=float)
    C = np.asarray(cfg[param_key]["C"], dtype=float)

    neuronal_params = BilinearParameters(A=A, B=B, C=C)
    if neuronal_params.l != l or neuronal_params.m != m:
        raise ValueError(
            f"Config model(l={l},m={m}) != neuronal params "
            f"(l={neuronal_params.l},m={neuronal_params.m})"
        )
    neuronal_model = BilinearNeuronalModel(neuronal_params)

    if cfg["hemodynamic"].get("use_defaults", True):
        hemo_params = HemodynamicParameters.with_defaults(l)
    else:
        raise NotImplementedError("Explicit hemodynamic params from YAML not implemented yet")

    hemo_model = HemodynamicBalloonModel(hemo_params)
    return ForwardModel(neuronal_model, hemo_model)
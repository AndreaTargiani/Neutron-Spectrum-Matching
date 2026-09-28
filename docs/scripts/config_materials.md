# `config_materials.py` — Material Library

## Purpose

These are the materials available for the experiment. `config_materials.py` is the single place they are defined: composition, density, and a bulk price. Every script that needs a material (`transfer_matrix.py`, `base_plates.py`, `optimizer.py`, `config_detector.py`, etc) imports it from here.

Which of them actually get \(S_N\) tensors, and which the optimizer is allowed to pick, are subsets declared in [`transfer_matrix.py`](transfer_matrix.md) and [`optimizer.py`](optimizer.md).

---

## Contents

| Section | API |
|---|---|
| [Shielding library](#shielding-library) | `build_material_library()` |
| [Air](#air) | `air_material()` |
| [Cost](#cost) | `COST_PER_KG` |

---

## Shielding library

Define each material with the usual OpenMC API (`openmc.Material`, `set_density`, `add_element` / `add_nuclide`). The dict **keys** are what you type everywhere else: plate lists, detector `material=`, optimizer search names.

Currently available: `concrete`, `graphite`, `HDPE`, `PE_BO`, `Pb`, `iron`, `cast_iron`, `SS_304`, `SS_316`, `water`, `tungsten`, `cadmium`, `B4C`, `Al_6061`, `Nickel`, `brass`, `Sulphur`, `Copper`, `Zirconium`, `SiC`, `PTFE`, `Mg`, `Bi`.

Their compositions come from the PNNL *Compendium of Material Composition Data for Radiation Transport Modeling* [[Detwiler et al., 2021]](#references).

!!! warning "Listed here does not mean the optimizer can pick it"
    `transfer_matrix.py` builds tensors only for its own `MATERIALS` subset. `optimizer.py` picks from its `MATERIALS` list, which can itself be a subset. Adding a material here does nothing in those scripts until you list it there too (and regenerate tensors).

### Adding a material

    1. Build an `openmc.Material` (density + nuclides / elements).
    2. Put it in the `return { ... }` dict under the key you want to type.
    3. Add the same key to [`COST_PER_KG`](#cost).

    ```python
    Inconel = openmc.Material(name="Inconel")
    Inconel.set_density("g/cm3", 8.44)
    Inconel.add_element("Ni", ...)
    # ...
    return {
        ...
        "Inconel": Inconel,
    }
    ```

    Then add `"Inconel"` to `transfer_matrix.py`'s `MATERIALS` (and run it) if you need tensors, and to `optimizer.py`'s `MATERIALS` if you want it in the search.

After changing a composition or density that already has tensors, regenerate the S_N database (`python src/transfer_matrix.py`). 

---

## Air

```python
def air_material() -> openmc.Material:
    ...
```

Dry NIST air (`name="air"`, 0.001205 g/cm³). It is not in the shielding library. `base_plates.py` uses it as the surrounding medium; [`config_detector.py`](config_detector.md) adds it to the detector lookup so a region can say `material="air"`.

---

## Cost

```python
COST_PER_KG: dict[str, float] = {
    "HDPE": ...,
    "tungsten": ...,
    ...
}
```

Put a price per kilogram (USD/kg) for each library key. It does not have to be what you will actually pay, though that would be ideal. If that number is unknown, the market price at which the material is traded is a good proxy.

How the optimizer turns this into a stack cost is on the [`optimizer.py`](optimizer.md#cost) page.

!!! warning "`COST_PER_KG` must cover every library key"
    The optimizer looks up `COST_PER_KG[mat]` for each layer. A new material with no price will fail there.

---

## References

- R. S. Detwiler, R. J. McConn, T. F. Grimes, S. A. Upton, and E. J. Engel, *Compendium of Material Composition Data for Radiation Transport Modeling*, PNNL-15870, Rev. 2 (200-DMAMC-128170), Pacific Northwest National Laboratory, April 2021. <https://doi.org/10.2172/1782721>

# Neutron Spectrum Matching via Bayesian Inference

Design the shielding that turns a known neutron source into a target spectrum, and see how sensitive that match is to each design parameter.

You set the source and the spectrum a detector should see. The code proposes materials, order, thicknesses, and plate size, then returns a range on each size. Spectra are compared after normalisation, so the match is the **shape**.

**[Documentation](https://AndreaTargiani.github.io/Neutron-Spectrum-Matching/)** · **[DEMO HCPB vacuum-vessel example](https://AndreaTargiani.github.io/Neutron-Spectrum-Matching/example/demo_vv/)**

## What it is for

A calibration often needs, on a bench, the spectrum an instrument will see in a machine. This framework does that design before any measurement. Transport is [OpenMC](https://openmc.org). The computational expensive steps have been made HPC compatible.

## A run, in order

1. **Configure** the source, the spectrum, the materials, and the detector.
2. **Build** the slab transfer database.
3. **Search** material order against minimizing both spectral error and cost.
4. **Check** the material order you choose with one 3-D (geometrically realistic) OpenMC run.
5. **Sample** the gap, the thicknesses, and the plate size.
6. **Calibrate** a surrogate and read off the posterior.

The [DEMO example](https://AndreaTargiani.github.io/Neutron-Spectrum-Matching/example/demo_vv/) walks through all six on the open FISPACT-II spectrum DEMO-HCPB-VV. The training set for that example is [`examples/demo_hcpb_vv.csv`](examples/demo_hcpb_vv.csv). It is a simple bench, and an easy place to start. The same steps still work for a much more complicated simulation and for other design parameters: the later steps only need the spectrum back on the working energy grid. Every setting is on the script pages.

## Install

- [OpenMC](https://docs.openmc.org/en/develop/usersguide/install.html), **develop** branch
- A cross-section library, for instance [ENDF/B-VIII.0 HDF5](https://openmc.org/data/)
- `pip install -r requirements.txt`

## License

See [LICENSE](LICENSE).

Master’s thesis work. The internship was partially funded by the FuseNet Association, with partial support from the EUROfusion Consortium.

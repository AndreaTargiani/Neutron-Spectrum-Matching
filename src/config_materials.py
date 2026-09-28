
from __future__ import annotations

import openmc


def build_material_library() -> dict[str, openmc.Material]:
    """Return a fresh ``{name: openmc.Material}`` shielding-material library."""
    concrete_nist = openmc.Material(name='CONCRETE_ordinary_NIST')
    concrete_nist.set_density('g/cm3', 2.3)
    concrete_nist.add_nuclide('H1', 0.305287, 'ao')
    concrete_nist.add_nuclide('H2', 0.000035, 'ao')
    concrete_nist.add_element('C', 0.002880, 'ao')
    concrete_nist.add_nuclide('O16', 0.499194, 'ao')
    concrete_nist.add_nuclide('O17', 0.000190, 'ao')
    concrete_nist.add_nuclide('O18', 0.001026, 'ao')
    concrete_nist.add_nuclide('Na23', 0.009212, 'ao')
    concrete_nist.add_element('Mg', 0.000725, 'ao')
    concrete_nist.add_nuclide('Al27', 0.010298, 'ao')
    concrete_nist.add_element('Si', 0.151046, 'ao')
    concrete_nist.add_element('K', 0.003578, 'ao')
    concrete_nist.add_element('Ca', 0.014924, 'ao')
    concrete_nist.add_element('Fe', 0.001605, 'ao')

    graphite = openmc.Material(name='graphite')
    graphite.set_density('g/cm3', 1.7)
    graphite.add_nuclide('B11', 0.000001, 'ao')
    graphite.add_element('C', 0.999999, 'ao')

    HDPE = openmc.Material(name='HDPE')
    HDPE.set_density('g/cm3', 0.95)
    HDPE.add_nuclide('H1', 0.666590, 'ao')
    HDPE.add_nuclide('H2', 0.000077, 'ao')
    HDPE.add_element('C', 0.333333, 'ao')

    PE_BO = openmc.Material(name='PE_BO')
    PE_BO.set_density('g/cm3', 1.0)
    PE_BO.add_nuclide('H1', 0.627683, 'ao')
    PE_BO.add_nuclide('H2', 0.000072, 'ao')
    PE_BO.add_nuclide('B10', 0.009289, 'ao')
    PE_BO.add_nuclide('B11', 0.037391, 'ao')
    PE_BO.add_element('C', 0.325564, 'ao')

    Pb = openmc.Material(name='Pb')
    Pb.set_density('g/cm3', 11.35)
    Pb.add_element('Pb', 1.0, 'ao')

    iron = openmc.Material(name='iron')
    iron.set_density('g/cm3', 7.874)
    iron.add_element('Fe', 1.0, 'ao')

    cast_iron = openmc.Material(name='cast_iron')
    cast_iron.set_density('g/cm3', 7.15)
    cast_iron.add_element('C', 0.137105, 'ao')
    cast_iron.add_element('Si', 0.044837, 'ao')
    cast_iron.add_nuclide('P31', 0.004691, 'ao')
    cast_iron.add_element('S', 0.001510, 'ao')
    cast_iron.add_nuclide('Mn55', 0.005730, 'ao')
    cast_iron.add_element('Fe', 0.806127, 'ao')

    SS_304 = openmc.Material(name='SS_304')
    SS_304.set_density('g/cm3', 8.03)
    SS_304.add_element('C', 0.003635, 'ao')
    SS_304.add_nuclide('Mn55', 0.019870, 'ao')
    SS_304.add_nuclide('P31', 0.000793, 'ao')
    SS_304.add_element('S', 0.000511, 'ao')
    SS_304.add_element('Si', 0.019434, 'ao')
    SS_304.add_element('Cr', 0.199443, 'ao')
    SS_304.add_element('Ni', 0.088343, 'ao')
    SS_304.add_element('Fe', 0.667971, 'ao')

    SS_316 = openmc.Material(name='SS_316')
    SS_316.set_density('g/cm3', 8.0)
    SS_316.add_element('C', 0.003683, 'ao')
    SS_316.add_nuclide('Mn55', 0.020128, 'ao')
    SS_316.add_nuclide('P31', 0.000803, 'ao')
    SS_316.add_element('S', 0.000517, 'ao')
    SS_316.add_element('Si', 0.019687, 'ao')
    SS_316.add_element('Cr', 0.180771, 'ao')
    SS_316.add_element('Ni', 0.113043, 'ao')
    SS_316.add_element('Mo', 0.014406, 'ao')
    SS_316.add_element('Fe', 0.646962, 'ao')

    water = openmc.Material(name='water')
    water.set_density('g/cm3', 0.997)
    water.add_nuclide('H1', 0.666590, 'ao')
    water.add_nuclide('H2', 0.000077, 'ao')
    water.add_nuclide('O16', 0.332523, 'ao')
    water.add_nuclide('O17', 0.000127, 'ao')
    water.add_nuclide('O18', 0.000683, 'ao')

    tungsten = openmc.Material(name='tungsten')
    tungsten.set_density('g/cm3', 19.3)
    tungsten.add_element('W', 1.0, 'ao')

    cadmium = openmc.Material(name='cadmium')
    cadmium.set_density('g/cm3', 8.65)
    cadmium.add_nuclide('Cd106', 0.012500, 'ao')
    cadmium.add_nuclide('Cd108', 0.008900, 'ao')
    cadmium.add_nuclide('Cd110', 0.124900, 'ao')
    cadmium.add_nuclide('Cd111', 0.128000, 'ao')
    cadmium.add_nuclide('Cd112', 0.241300, 'ao')
    cadmium.add_nuclide('Cd113', 0.122200, 'ao')
    cadmium.add_nuclide('Cd114', 0.287300, 'ao')
    cadmium.add_nuclide('Cd116', 0.074900, 'ao')

    B4C = openmc.Material(name='B4C')
    B4C.set_density('g/cm3', 2.52)
    B4C.add_nuclide('B10', 0.1592, 'ao')
    B4C.add_nuclide('B11', 0.6408, 'ao')
    B4C.add_element('C', 0.2000, 'ao')

    Al_6061 = openmc.Material(name='Al_6061')
    Al_6061.set_density('g/cm3', 2.70)
    Al_6061.add_element('Mg', 0.011162, 'ao')
    Al_6061.add_nuclide('Al27', 0.977330, 'ao')
    Al_6061.add_element('Si', 0.005796, 'ao')
    Al_6061.add_element('Ti', 0.000496, 'ao')
    Al_6061.add_element('Cr', 0.001017, 'ao')
    Al_6061.add_nuclide('Mn55', 0.000433, 'ao')
    Al_6061.add_element('Fe', 0.001986, 'ao')
    Al_6061.add_element('Cu', 0.001174, 'ao')
    Al_6061.add_element('Zn', 0.000606, 'ao')

    Nickel = openmc.Material(name='Nickel')
    Nickel.set_density('g/cm3', 8.902)
    Nickel.add_element('Ni', 1.0, 'ao')

    brass = openmc.Material(name='brass')
    brass.set_density('g/cm3', 8.07)
    brass.add_element('Fe', 0.000992, 'ao')
    brass.add_element('Cu', 0.668462, 'ao')
    brass.add_element('Zn', 0.318026, 'ao')
    brass.add_element('Sn', 0.001437, 'ao')
    brass.add_nuclide('P31', 0.011083, 'ao')

    Sulphur = openmc.Material(name='Sulphur')
    Sulphur.set_density('g/cm3', 2.0)
    Sulphur.add_element('S', 1.0, 'ao')

    Copper = openmc.Material(name='Copper')
    Copper.set_density('g/cm3', 8.96)
    Copper.add_element('Cu', 1.0, 'ao')

    Zirconium = openmc.Material(name='Zirconium')
    Zirconium.set_density('g/cm3', 6.52)
    Zirconium.add_nuclide('Zr90', 0.5145, 'ao')
    Zirconium.add_nuclide('Zr91', 0.1122, 'ao')
    Zirconium.add_nuclide('Zr92', 0.1715, 'ao')
    Zirconium.add_nuclide('Zr94', 0.1738, 'ao')
    Zirconium.add_nuclide('Zr96', 0.0280, 'ao')

    SiC = openmc.Material(name='SiC')
    SiC.set_density('g/cm3', 3.21)
    SiC.add_element('Si', 0.5, 'ao')
    SiC.add_element('C', 0.5, 'ao')

    PTFE = openmc.Material(name='PTFE')
    PTFE.set_density('g/cm3', 2.25)
    PTFE.add_element('C', 0.333333, 'ao')
    PTFE.add_nuclide('F19', 0.666667, 'ao')

    Mg = openmc.Material(name='Mg')
    Mg.set_density('g/cm3', 1.74)
    Mg.add_element('Mg', 1.0, 'ao')

    Bi = openmc.Material(name='Bi')
    Bi.set_density('g/cm3', 9.747)
    Bi.add_nuclide('Bi209', 1.0, 'ao')

    return {
        'concrete': concrete_nist, 'graphite': graphite, 'HDPE': HDPE,
        'PE_BO': PE_BO, 'Pb': Pb, 'iron': iron, 'cast_iron': cast_iron,
        'SS_304': SS_304, "SS_316": SS_316, 'water': water, 'tungsten': tungsten,
        'cadmium': cadmium, 'B4C': B4C, 'Al_6061': Al_6061,
        'Nickel': Nickel, 'brass': brass, 'Sulphur': Sulphur,
        'Copper': Copper, 'Zirconium': Zirconium, 'SiC': SiC,
        'PTFE': PTFE, 'Mg': Mg, 'Bi': Bi,
    }


def air_material() -> openmc.Material:
    """Return the ambient air ``openmc.Material`` (dry, NIST composition)."""
    air = openmc.Material(name='air')
    air.set_density('g/cm3', 0.001205)
    air.add_element('C',    0.000150, 'ao')
    air.add_nuclide('N14',  0.781574, 'ao')
    air.add_nuclide('N15',  0.002855, 'ao')
    air.add_nuclide('O16',  0.210238, 'ao')
    air.add_nuclide('O17',  0.000080, 'ao')
    air.add_nuclide('O18',  0.000432, 'ao')
    air.add_element('Ar',   0.004671, 'ao')
    return air


# Indicative bulk material prices [USD/kg]; 
COST_PER_KG: dict[str, float] = {
    'concrete': 0.15, #USGS
    'graphite': 3, #https://gotraysgraphite.com/graphite-block-and-graphite-powder-price/
    'HDPE': 1.2, #https://www.expertmarketresearch.com/price-forecast/hdpe-price-forecast
    'PE_BO': 1.4, #x1.2 HDPE, not common
    'Pb': 1.9, #https://tradingeconomics.com/commodity/lead
    'iron': 1.5, # https://www.materialpricebook.com/prices/steel/a36/plate, but also according to USGS
    'cast_iron': 2.2, #x1.5 iron, not common
    'SS_304': 3.0, #https://businessanalytiq.com/procurementanalytics/index/stainless-steel-plate-index/
    'SS_316': 3.6, #x1.2 SS304
    'water': 0.01,
    'tungsten': 250, #USGS
    'cadmium': 3,#https://www.expertmarketresearch.com/price-forecast/cadmium-price-trends
    'B4C': 50, #https://www.made-in-china.com/products-search/hot-china-products/Boron_Carbide_Price.html
    'Al_6061': 3, #USGS
    'Nickel': 17, #USGS
    'brass': 10.6, #USGS
    'Sulphur': 0.15, #place holder
    'Copper': 14.5, #USGS
    'Zirconium': 300.0, #place holder
    'SiC': 300.0, #place holder
    'PTFE': 12.7, #https://businessanalytiq.com/procurementanalytics/index/polytetrafluoroethylene-ptfe-price-index/
    'Mg': 7.9, #USGS
    'Bi': 12, #USGS
}

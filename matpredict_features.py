from pathlib import Path
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def configurar_columnas(raiz):
    global FEATURES, CAT
    ruta=Path(raiz)/'data'/'processed_clasificacion'/'manifest_clasificacion.json'
    if not ruta.exists():
        raise FileNotFoundError(f'No se encontró el manifiesto de entrenamiento: {ruta}')
    info=json.loads(ruta.read_text(encoding='utf-8'))
    FEATURES=info['columnas_predictoras']
    CAT=info['columnas_categoricas']
    if len(FEATURES)!=46:
        raise ValueError(f'Se esperaban 46 características; hay {len(FEATURES)}')
    return FEATURES

FEATURES=[]
CAT=[]

PROPIEDADES={
 'Z':lambda e:e.Z,
 'masa_atomica':lambda e:float(e.atomic_mass),
 'electronegatividad':lambda e:e.X,
 'fila_periodica':lambda e:e.row,
 'grupo_periodico':lambda e:e.group,
 'radio_atomico':lambda e:e.atomic_radius,
}
def a_float(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError,ValueError,OverflowError):
        return np.nan

def composicion_a_features(formula):
    comp=Composition(str(formula))
    fracciones=comp.fractional_composition.get_el_amt_dict()
    elementos=[Element(k) for k in fracciones]
    pesos=np.array([fracciones[e.symbol] for e in elementos],dtype=float)
    pesos=pesos/pesos.sum()
    salida={'comp_num_elementos':len(elementos),
            'comp_entropia_mezcla':float(-np.sum(pesos*np.log(pesos)))}
    for nombre,func in PROPIEDADES.items():
        valores=np.array([a_float(func(e)) for e in elementos],dtype=float)
        validos=np.isfinite(valores)
        for est in ('media','min','max','desv','rango'):
            salida[f'comp_{nombre}_{est}']=np.nan
        if validos.any():
            v=valores[validos]; w=pesos[validos]/pesos[validos].sum()
            media=float(np.dot(v,w))
            salida[f'comp_{nombre}_media']=media
            salida[f'comp_{nombre}_min']=float(v.min())
            salida[f'comp_{nombre}_max']=float(v.max())
            salida[f'comp_{nombre}_desv']=float(np.sqrt(np.dot(w,(v-media)**2)))
            salida[f'comp_{nombre}_rango']=float(v.max()-v.min())
    return salida

print('Descriptores químicos:',len(composicion_a_features('NaCl')))
ESTRUCTURA_NUM=['nelements','nsites','density','volume','volume_per_atom',
 'atomic_number_density','space_group_number','lattice_a','lattice_b','lattice_c',
 'lattice_alpha','lattice_beta','lattice_gamma']
ESTRUCTURA_CAT=['crystal_system']
SISTEMAS={'triclinic':'Triclinic','monoclinic':'Monoclinic','orthorhombic':'Orthorhombic',
 'tetragonal':'Tetragonal','trigonal':'Trigonal','hexagonal':'Hexagonal','cubic':'Cubic'}

def estructura_a_features(estructura, symprec=0.01, angle_tolerance=5):
    if not isinstance(estructura,Structure):
        raise TypeError('Se requiere un objeto pymatgen Structure.')
    if not estructura.is_ordered:
        raise ValueError('El CIF contiene ocupaciones parciales o desorden; no se admite automáticamente.')
    if len(estructura)==0 or estructura.volume<=0:
        raise ValueError('Estructura vacía o volumen inválido.')
    lat=estructura.lattice
    analizador=SpacegroupAnalyzer(estructura,symprec=symprec,angle_tolerance=angle_tolerance)
    sistema=analizador.get_crystal_system()
    if sistema not in SISTEMAS: raise ValueError(f'Sistema cristalino desconocido: {sistema}')
    return {
        'nelements':len(estructura.composition.elements),
        'nsites':len(estructura),
        'density':float(estructura.density),
        'volume':float(estructura.volume),
        'volume_per_atom':float(estructura.volume/len(estructura)),
        'atomic_number_density':float(len(estructura)/estructura.volume*1e24),
        'space_group_number':int(analizador.get_space_group_number()),
        'lattice_a':float(lat.a),'lattice_b':float(lat.b),'lattice_c':float(lat.c),
        'lattice_alpha':float(lat.alpha),'lattice_beta':float(lat.beta),'lattice_gamma':float(lat.gamma),
        'crystal_system':SISTEMAS[sistema],
    }
print('Descriptores estructurales:',len(ESTRUCTURA_NUM)+len(ESTRUCTURA_CAT))
def _composiciones_compatibles(a,b,rtol=1e-5,atol=1e-8):
    da=Composition(a).fractional_composition.get_el_amt_dict()
    db=Composition(b).fractional_composition.get_el_amt_dict()
    return set(da)==set(db) and all(np.isclose(da[e],db[e],rtol=rtol,atol=atol) for e in da)

def extraer_descriptores_cif(ruta_cif,formula=None,symprec=0.01):
    ruta=Path(ruta_cif)
    if not ruta.is_file(): raise FileNotFoundError(f'No existe el CIF: {ruta}')
    if ruta.suffix.lower()!='.cif': raise ValueError('Se requiere un archivo .cif')
    estructura=Structure.from_file(str(ruta))
    if not estructura.is_ordered:
        raise ValueError('Estructura desordenada: revisar ocupaciones parciales antes de inferir.')
    formula_cif=estructura.composition.reduced_formula
    if formula is not None and str(formula).strip():
        if not _composiciones_compatibles(str(formula).strip(),estructura.composition):
            raise ValueError(f'La fórmula {formula!r} no coincide con la composición del CIF ({formula_cif}).')
    features={**composicion_a_features(formula_cif),**estructura_a_features(estructura,symprec=symprec)}
    faltan=set(FEATURES)-set(features)
    extras=set(features)-set(FEATURES)
    if faltan or extras: raise AssertionError(f'Desajuste con entrenamiento. Faltan={faltan}; extras={extras}')
    X=pd.DataFrame([features],columns=FEATURES)
    for col in FEATURES:
        if col not in CAT: X[col]=pd.to_numeric(X[col],errors='coerce')
    X=X.replace([np.inf,-np.inf],np.nan)
    meta={'archivo':ruta.name,'formula_cif':formula_cif,'formula_usuario':formula,
          'symprec':symprec,'sitios':len(estructura),'volumen_A3':float(estructura.volume)}
    return X,meta

print('Extractor preparado.')

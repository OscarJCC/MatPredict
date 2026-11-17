from pathlib import Path
import tempfile
import json
import joblib
import pandas as pd
import streamlit as st
from matpredict_features import configurar_columnas, extraer_descriptores_cif, composicion_a_features

st.set_page_config(page_title='MatPredict',page_icon='🧪',layout='wide')
ROOT=Path(__file__).resolve().parent
DEPLOY_CIF=ROOT/'deployment'/'matpredict_v1'
DEPLOY_FORMULA=ROOT/'deployment'/'matpredict_formula_v1'

@st.cache_resource(show_spinner='Cargando modelos de MatPredict...')
def cargar_modelos(modalidad):
    if modalidad=='formula':
        contrato=json.loads((DEPLOY_FORMULA/'contrato_formula.json').read_text(encoding='utf-8'))
        columnas=contrato['columnas_predictoras']
        if len(columnas)!=32 or any(not x.startswith('comp_') for x in columnas):
            raise ValueError('Contrato de fórmula incompatible: se esperan 32 descriptores comp_.')
        base=DEPLOY_FORMULA
        archivos=['modelo_metalica_formula.joblib','modelo_magnetica_formula.joblib']
    else:
        columnas=configurar_columnas(ROOT)
        base=DEPLOY_CIF
        archivos=['modelo_metalica.joblib','modelo_magnetica.joblib']
    modelos=[joblib.load(base/f) for f in archivos] # Solo archivos propios y confiables
    for modelo in modelos:
        if hasattr(modelo,'feature_names_in_') and list(modelo.feature_names_in_)!=list(columnas):
            raise ValueError('El orden de columnas no coincide con el modelo entrenado.')
    return modelos[0],modelos[1],columnas

def score_positivo(modelo,X):
    clases=list(modelo.classes_)
    idx=next((i for i,c in enumerate(clases) if str(c).lower() in ('1','true')),None)
    if idx is None: raise ValueError(f'Clase positiva ausente: {clases}')
    return float(modelo.predict_proba(X)[0,idx])

def predecir(modalidad,X):
    m,g,columnas=cargar_modelos(modalidad)
    X=X.loc[:,columnas]
    return score_positivo(m,X),score_positivo(g,X)

def mostrar_resultados(titulo,formula,X,pm,pg):
    st.subheader(titulo)
    metal=pm>=0.5
    magnetico=pg>=0.5
    a,b=st.columns(2)
    with a:
        st.metric('Metalicidad','Metálico' if metal else 'No metálico')
        st.progress(pm,text=f'Score no calibrado: {pm:.4f}')
    with b:
        st.metric('Magnetismo','Magnético' if magnetico else 'No magnético')
        st.progress(pg,text=f'Score no calibrado: {pg:.4f}')
    with st.expander(f'Ver los {len(X.columns)} descriptores utilizados'):
        st.dataframe(X.T.rename(columns={0:'Valor'}),use_container_width=True)
    return {'modalidad':titulo,'formula':formula,'score_metalico_no_calibrado':pm,
            'pred_is_metal':int(metal),'score_magnetico_no_calibrado':pg,
            'pred_is_magnetic':int(magnetico),'umbral_metalico':0.5,'umbral_magnetico':0.5}

st.title('🧪 MatPredict')
st.write('Predicción exploratoria de metalicidad y magnetismo mediante composición química o estructura cristalina.')
st.info('Los scores no están calibrados: no equivalen a certeza física. Las predicciones no sustituyen cálculos DFT ni validación experimental.')
modo=st.radio('Selecciona la modalidad', ['Solo fórmula química (32 descriptores)','Estructura CIF (46 descriptores)'],horizontal=True)
con_cif=modo.startswith('Estructura')
with st.form('form_material'):
    formula=st.text_input('Fórmula química' + (' (opcional, para verificar el CIF)' if con_cif else ' (obligatoria)'),placeholder='NaCl, Fe2O3, Nb4C3')
    archivo=st.file_uploader('Archivo CIF',type=['cif']) if con_cif else None
    comparar=st.checkbox('Comparar también con los modelos de solo fórmula',value=False) if con_cif else False
    enviar=st.form_submit_button('Analizar material',type='primary')

if enviar:
    try:
        if con_cif:
            if archivo is None:
                st.warning('Debes cargar un CIF para esta modalidad.');st.stop()
            if archivo.size>5*1024*1024:
                st.error('El CIF supera el límite de 5 MB.');st.stop()
            with tempfile.TemporaryDirectory(prefix='matpredict_') as carpeta:
                ruta=Path(carpeta)/'entrada.cif';ruta.write_bytes(archivo.getvalue())
                configurar_columnas(ROOT)
                X,meta=extraer_descriptores_cif(ruta,formula=formula.strip() or None)
            formula_final=meta['formula_cif']
            st.write(f"**Fórmula del CIF:** {formula_final} · **Sitios:** {meta['sitios']} · **Volumen:** {meta['volumen_A3']:.4f} Å³")
            pm,pg=predecir('cif',X)
            resultados=[mostrar_resultados('Modelo con CIF (46 descriptores)',formula_final,X,pm,pg)]
            if comparar:
                _,_,columnas=cargar_modelos('formula')
                Xf=pd.DataFrame([composicion_a_features(formula_final)],columns=columnas)
                pmf,pgf=predecir('formula',Xf)
                resultados.append(mostrar_resultados('Modelo solo fórmula (32 descriptores)',formula_final,Xf,pmf,pgf))
                st.caption('La comparación refleja dos modelos distintos; la diferencia no es una estimación de incertidumbre calibrada.')
        else:
            if not formula.strip():
                st.warning('Introduce una fórmula química válida.');st.stop()
            _,_,columnas=cargar_modelos('formula')
            X=pd.DataFrame([composicion_a_features(formula.strip())],columns=columnas)
            pm,pg=predecir('formula',X)
            resultados=[mostrar_resultados('Modelo solo fórmula (32 descriptores)',formula.strip(),X,pm,pg)]
        st.subheader('Orientaciones preliminares para Quantum ESPRESSO')
        metal=resultados[0]['pred_is_metal']==1
        magnetico=resultados[0]['pred_is_magnetic']==1
        st.write("**Ocupaciones:** explorar `occupations='smearing'` y convergencia del smearing." if metal else "**Ocupaciones:** evaluar `occupations='fixed'` y verificar el comportamiento electrónico mediante DFT.")
        st.write('**Espín:** explorar `nspin=2`, magnetizaciones iniciales y posibles ordenamientos.' if magnetico else '**Espín:** considerar `nspin=1` como prueba inicial, sin descartar configuraciones spin-polarizadas.')
        if not con_cif:
            st.warning('Sin CIF no se conocen la estructura ni el polimorfo. Una misma fórmula puede tener propiedades diferentes según su fase.')
        else:
            st.warning('Un CIF experimental o no relajado puede diferir de las estructuras usadas en entrenamiento.')
        st.download_button('Descargar resultados CSV',pd.DataFrame(resultados).to_csv(index=False).encode('utf-8'),file_name='matpredict_predicciones.csv',mime='text/csv')
    except (ValueError,FileNotFoundError,TypeError,KeyError,AssertionError) as exc:
        st.error(f'No fue posible analizar el material: {exc}')
    except Exception as exc:
        st.error(f'Error durante el análisis ({type(exc).__name__}): {exc}')
st.divider()
st.caption('MatPredict · Prototipo de investigación · Umbrales 0.50 · No predice ecutwfc, malla k ni parámetros Hubbard U.')

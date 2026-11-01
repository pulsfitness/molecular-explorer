

import streamlit as st
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    Draw,
    Crippen,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdDepictor


# ==========================================
# 1. CONFIGURARE
# ==========================================

st.set_page_config(
    page_title="Molecular Scaffold Explorer",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 750;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #8A94A6;
        font-size: 15px;
    }
    div[data-testid="stMetric"] {
        background: rgba(120, 130, 150, 0.08);
        padding: 14px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🧬 Molecular Scaffold Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Platformă de analiză structurală și comparație moleculară</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 2. CONFIGURAREA ANALIZEI
# ==========================================

DEFAULT_SMILES = "CNCCOCC(=O)Nc1ccc(Cl)cc1"

PATTERNS = {
    "Amidă": "[CX3](=O)[NX3]",
    "Eter": "[OD2]([#6])[#6]",
    "Amină primară": "[NX3;H2;!$(NC=O)]",
    "Amină secundară": "[NX3;H1;!$(NC=O)]",
    "Amină terțiară": "[NX3;H0;!$(NC=O)]",
    "Inel aromatic": "a1aaaaa1",
    "Halogen": "[F,Cl,Br,I]",
    "Nitril": "[C]#N",
    "Sulfonamidă": "[SX4](=O)(=O)[NX3]",
    "Uree": "[NX3][CX3](=O)[NX3]",
    "Ester": "[CX3](=O)[OX2H0][#6]",
    "Acid carboxilic": "[CX3](=O)[OX2H1]",
    "Fenol": "[OX2H][c]",
}


def parse_smiles(smiles):
    """Transformă un SMILES într-o moleculă RDKit."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def detect_groups(mol):
    """Identifică motive structurale folosind reguli SMARTS."""
    found = []

    for name, smarts in PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)

        if pattern and mol.HasSubstructMatch(pattern):
            found.append(name)

    return found


def get_scaffold(mol):
    """Extrage scaffold-ul Bemis-Murcko."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold.GetNumAtoms() == 0:
        return scaffold, "Nu există scaffold ciclic identificabil."

    return scaffold, Chem.MolToSmiles(scaffold)


def get_descriptors(mol):
    """Calculează descriptorii moleculari."""
    return {
        "Masă moleculară (Da)": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Donori H": rdMolDescriptors.CalcNumHBD(mol),
        "Acceptori H": rdMolDescriptors.CalcNumHBA(mol),
        "Legături rotative": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Număr de inele": rdMolDescriptors.CalcNumRings(mol),
        "Fracție C sp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol), 3
        ),
        "Atomi grei": mol.GetNumHeavyAtoms(),
    }


# Amprentă Morgan pentru comparația moleculară
morgan_generator = GetMorganGenerator(
    radius=2,
    fpSize=2048
)


# ==========================================
# 3. MENIU
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "🧪 Analiză moleculară",
    "🔬 Comparație moleculară",
    "📘 Metodologie"
])


# ==========================================
# 4. ANALIZA UNEI MOLECULE
# ==========================================

with tab1:

    st.header("Analiza structurii moleculare")

    smiles = st.text_area(
        "Introdu structura SMILES:",
        value=DEFAULT_SMILES,
        height=90
    )

    analyze = st.button(
        "🔍 ANALIZEAZĂ MOLECULA",
        type="primary",
        key="analyze"
    )

    if analyze:

        mol = parse_smiles(smiles)

        if mol is None:
            st.error("SMILES invalid. Verifică structura introdusă.")

        else:

            canonical = Chem.MolToSmiles(mol)
            scaffold, scaffold_smiles = get_scaffold(mol)
            descriptors = get_descriptors(mol)
            groups = detect_groups(mol)

            st.success("Analiza structurală a fost realizată cu succes.")

            st.subheader("1. Reprezentarea moleculară")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    Draw.MolToImage(mol, size=(600, 400)),
                    caption="Structura moleculară 2D",
                    use_container_width=True
                )

            with col2:
                st.markdown("**SMILES canonic**")
                st.code(canonical)

                st.markdown("**Scaffold Bemis–Murcko**")
                st.code(scaffold_smiles)

                if scaffold.GetNumAtoms() > 0:
                    st.image(
                        Draw.MolToImage(scaffold, size=(500, 350)),
                        caption="Scheletul molecular extras",
                        use_container_width=True
                    )

            st.divider()

            st.subheader("2. Proprietăți fizico-chimice")

            metric_columns = st.columns(3)

            for i, (name, value) in enumerate(descriptors.items()):
                metric_columns[i % 3].metric(name, value)

            st.divider()

            st.subheader("3. Grupări funcționale")

            if groups:
                group_columns = st.columns(3)

                for i, group in enumerate(groups):
                    group_columns[i % 3].success("✓ " + group)
            else:
                st.warning(
                    "Nu au fost detectate motive din lista de reguli implementate."
                )

            st.divider()

            st.subheader("4. Raportul analizei")

            report = pd.DataFrame(
                list(descriptors.items()),
                columns=["Parametru", "Valoare"]
            )

            report.loc[len(report)] = [
                "SMILES canonic", canonical
            ]

            report.loc[len(report)] = [
                "Scaffold Bemis–Murcko", scaffold_smiles
            ]

            report.loc[len(report)] = [
                "Grupări detectate", ", ".join(groups)
            ]

            st.download_button(
                "⬇️ Descarcă raportul CSV",
                data=report.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="raport_molecular.csv",
                mime="text/csv"
            )


# ==========================================
# 5. COMPARAȚIE MOLECULARĂ
# ==========================================

with tab2:

    st.header("Comparație între două molecule")

    st.write(
        "Compară două structuri moleculare folosind "
        "amprente Morgan și coeficientul de similaritate Tanimoto."
    )

    col1, col2 = st.columns(2)

    with col1:
        smiles_a = st.text_area(
            "Molecula A",
            value=DEFAULT_SMILES,
            height=100
        )

    with col2:
        smiles_b = st.text_area(
            "Molecula B",
            value="CC(=O)NC1=CC=C(O)C=C1",
            height=100
        )

    compare = st.button(
        "⚖️ COMPARĂ MOLECULELE",
        type="primary",
        key="compare"
    )

    if compare:

        mol_a = parse_smiles(smiles_a)
        mol_b = parse_smiles(smiles_b)

        if mol_a is None or mol_b is None:

            st.error("Una sau ambele structuri SMILES sunt invalide.")

        else:

            fp_a = morgan_generator.GetFingerprint(mol_a)
            fp_b = morgan_generator.GetFingerprint(mol_b)

            similarity = DataStructs.TanimotoSimilarity(
                fp_a, fp_b
            )

            st.success("Comparația a fost finalizată.")

            st.metric(
                "Similaritate structurală Tanimoto",
                f"{similarity * 100:.2f}%"
            )

            st.progress(similarity)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Molecula A**")
                st.image(
                    Draw.MolToImage(mol_a, size=(450, 300)),
                    use_container_width=True
                )

            with col2:
                st.markdown("**Molecula B**")
                st.image(
                    Draw.MolToImage(mol_b, size=(450, 300)),
                    use_container_width=True
                )

            st.subheader("Interpretarea comparației")

            st.write(
                f"Coeficientul Tanimoto calculat este "
                f"{similarity:.4f}, utilizând amprente Morgan "
                f"cu raza 2 și 2048 de biți."
            )

            st.info(
                "Similaritatea structurală nu demonstrează "
                "că moleculele au aceeași activitate biologică, "
                "aceeași toxicitate sau același mecanism de acțiune."
            )

            comparison_report = pd.DataFrame({
                "Parametru": [
                    "SMILES molecula A",
                    "SMILES molecula B",
                    "Metodă",
                    "Rază Morgan",
                    "Dimensiune amprentă",
                    "Similaritate Tanimoto"
                ],
                "Valoare": [
                    Chem.MolToSmiles(mol_a),
                    Chem.MolToSmiles(mol_b),
                    "Morgan fingerprint",
                    2,
                    2048,
                    round(similarity, 4)
                ]
            })

            st.download_button(
                "⬇️ Descarcă raportul comparației",
                data=comparison_report.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="comparatie_moleculara.csv",
                mime="text/csv"
            )


# ==========================================
# 6. METODOLOGIE
# ==========================================

with tab3:

    st.header("Metodologia algoritmului")

    st.markdown("""
    ### Biblioteci utilizate

    - **RDKit:** procesarea și analiza structurilor chimice.
    - **Streamlit:** interfața grafică.
    - **Pandas:** organizarea și exportul rezultatelor.

    ### Metode computaționale

    **1. Reprezentarea moleculară**

    Structurile SMILES sunt transformate în obiecte moleculare
    pentru analiza ulterioară.

    **2. Descriptori moleculari**

    Sunt calculați parametri precum masa moleculară, LogP,
    TPSA și numărul donorilor și acceptorilor de hidrogen.

    **3. Extragerea scaffold-ului**

    Se utilizează metoda Bemis–Murcko pentru identificarea
    scheletului molecular.

    **4. Detectarea grupărilor funcționale**

    Se utilizează reguli SMARTS pentru recunoașterea
    unor motive structurale.

    **5. Similaritatea moleculară**

    Se utilizează amprente Morgan și coeficientul Tanimoto:

    Tanimoto(A,B) = c / (a + b - c)

    unde a și b reprezintă numărul biților activi în fiecare
    amprentă, iar c reprezintă numărul biților comuni.
    """)

    st.warning(
        "Metodele oferă rezultate computaționale exploratorii. "
        "Identificarea motivelor structurale și interpretarea "
        "chimică necesită verificare. Similaritatea nu reprezintă "
        "dovada activității biologice."
    )


# ==========================================
# 7. SUBSOL
# ==========================================

st.divider()

st.caption(
    "Molecular Scaffold Explorer | Instrument educațional "
    "de chimioinformatică | Analiză realizată cu RDKit"
)

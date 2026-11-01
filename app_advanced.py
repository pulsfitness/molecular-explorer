
import streamlit as st
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator


# ---------------------------------------------------------
# CONFIGURAREA PAGINII
# ---------------------------------------------------------

st.set_page_config(
    page_title="Molecular Explorer",
    page_icon="🧪",
    layout="wide"
)

# ---------------------------------------------------------
# IDENTITATEA INSTITUTIONALA
# Folosim componente Streamlit native pentru ca textul
# sa fie vizibil si fara HTML sau CSS.
# ---------------------------------------------------------

st.title("COLEGIUL TEHNIC „ANA ASLAN” CLUJ-NAPOCA")
st.subheader("Județul Cluj")

st.markdown("---")

st.title("🧪 Molecular Explorer")
st.markdown(
    "### Platformă experimentală de analiză chimioinformatică"
)

st.markdown(
    """
    **Realizat în scop experimental de eleva Apahidean Maria Giulia**  
    **Sub îndrumarea profesoarei coordonatoare Chicinaș Raluca**  
    **An școlar: 2026–2027**
    """
)

st.markdown("---")


# ---------------------------------------------------------
# FUNCTII PENTRU ANALIZA MOLECULELOR
# ---------------------------------------------------------

DEFAULT_SMILES = "CNCCOCC(=O)Nc1ccc(cc1)Cl"


def parse_molecule(smiles):
    """Transforma un sir SMILES intr-o molecula RDKit."""
    if not smiles or not smiles.strip():
        return None

    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def calculate_descriptors(mol):
    """Calculeaza descriptorii moleculari principali."""
    return {
        "Formula moleculară": rdMolDescriptors.CalcMolFormula(mol),
        "Masă moleculară (g/mol)": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Donori de legături de hidrogen": Lipinski.NumHDonors(mol),
        "Acceptori de legături de hidrogen": Lipinski.NumHAcceptors(mol),
        "Legături rotative": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Număr de atomi grei": mol.GetNumHeavyAtoms(),
        "Număr de inele": rdMolDescriptors.CalcNumRings(mol),
    }


# Modele SMARTS orientative pentru identificarea unor grupe.
# O grupă poate fi identificata de mai multe ori.
FUNCTIONAL_GROUPS = {
    "Grupă hidroxil": "[OX2H]",
    "Grupă carbonil": "[CX3]=[OX1]",
    "Grupă amidă": "[NX3][CX3](=[OX1])",
    "Grupă amină": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Grupă eter": "[OD2]([#6])[#6]",
    "Halogen": "[F,Cl,Br,I]",
    "Inel aromatic": "a1aaaaa1",
    "Grupă nitril": "[CX2]#N",
}


def find_functional_groups(mol):
    """Returneaza numarul de potriviri pentru fiecare SMARTS."""
    results = {}

    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)

        if pattern is None:
            results[name] = 0
        else:
            results[name] = len(mol.GetSubstructMatches(pattern))

    return results


def get_scaffold(mol):
    """Extrage scheletul Bemis-Murcko al moleculei."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None

    return scaffold


def molecule_image(mol, size=(500, 350)):
    """Genereaza imaginea 2D a moleculei."""
    mol_copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol_copy)
    return Draw.MolToImage(mol_copy, size=size)


def create_fingerprint(mol):
    """Genereaza amprenta Morgan pentru comparatie."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )
    return generator.GetFingerprint(mol)


# ---------------------------------------------------------
# TAB-URI
# ---------------------------------------------------------

tab_analysis, tab_comparison, tab_methodology = st.tabs(
    [
        "🔬 Analiza moleculei",
        "⚖️ Comparație moleculară",
        "📚 Metodologie"
    ]
)


# =========================================================
# TAB 1 - ANALIZA MOLECULEI
# =========================================================

with tab_analysis:

    st.header("Analiza structurii moleculare")

    smiles = st.text_input(
        "Introdu secvența SMILES:",
        value=DEFAULT_SMILES,
        help="SMILES reprezintă o modalitate textuală de descriere a structurii chimice."
    )

    if st.button("Analizează molecula", type="primary"):
        mol = parse_molecule(smiles)

        if mol is None:
            st.error(
                "Secvența SMILES nu este validă. "
                "Verifică textul introdus și încearcă din nou."
            )
        else:
            st.session_state["current_smiles"] = smiles
            st.session_state["analysis_done"] = True

    # Afisam analiza dupa apasarea butonului.
    if st.session_state.get("analysis_done", False):

        active_smiles = st.session_state.get(
            "current_smiles",
            DEFAULT_SMILES
        )
        mol = parse_molecule(active_smiles)

        if mol is not None:

            st.success("Molecula a fost identificată cu succes.")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Structura 2D")
                st.image(
                    molecule_image(mol),
                    caption="Reprezentarea bidimensională a moleculei",
                    use_container_width=True
                )

            with col2:
                st.subheader("Informații generale")

                canonical_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True
                )

                st.write("**SMILES canonic:**")
                st.code(canonical_smiles)

                st.write("**Număr de atomi:**", mol.GetNumAtoms())
                st.write(
                    "**Număr de atomi grei:**",
                    mol.GetNumHeavyAtoms()
                )

                st.write(
                    "**Număr de legături:**",
                    mol.GetNumBonds()
                )

            st.markdown("---")

            # DESCRIPTORI
            st.subheader("Descriptorii moleculari")

            descriptors = calculate_descriptors(mol)

            descriptor_df = pd.DataFrame(
                list(descriptors.items()),
                columns=["Descriptor", "Valoare"]
            )

            st.dataframe(
                descriptor_df,
                use_container_width=True,
                hide_index=True
            )

            csv_descriptors = descriptor_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Descarcă descriptorii în format CSV",
                data=csv_descriptors,
                file_name="descriptorii_moleculei.csv",
                mime="text/csv"
            )

            st.markdown("---")

            # GRUPE FUNCTIONALE
            st.subheader("Identificarea grupelor funcționale")

            functional_groups = find_functional_groups(mol)

            groups_df = pd.DataFrame(
                [
                    {
                        "Grupă funcțională": name,
                        "Număr de apariții": count
                    }
                    for name, count in functional_groups.items()
                ]
            )

            st.dataframe(
                groups_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Identificarea se bazează pe tipare SMARTS. "
                "Rezultatele sunt orientative și depind de regulile utilizate."
            )

            st.markdown("---")

            # SCAFFOLD
            st.subheader("Scheletul molecular Bemis–Murcko")

            scaffold = get_scaffold(mol)

            if scaffold is not None:

                scaffold_col1, scaffold_col2 = st.columns(2)

                with scaffold_col1:
                    st.image(
                        molecule_image(scaffold, size=(400, 300)),
                        caption="Schelet molecular Bemis–Murcko",
                        use_container_width=True
                    )

                with scaffold_col2:
                    scaffold_smiles = Chem.MolToSmiles(scaffold)

                    st.write("**SMILES al scheletului:**")
                    st.code(scaffold_smiles)

                    st.write(
                        "**Număr de atomi în schelet:**",
                        scaffold.GetNumAtoms()
                    )

                    st.write(
                        "**Număr de inele în schelet:**",
                        rdMolDescriptors.CalcNumRings(scaffold)
                    )

            else:
                st.info(
                    "Nu a fost identificat un schelet Bemis–Murcko "
                    "nevid pentru această moleculă."
                )


# =========================================================
# TAB 2 - COMPARATIE MOLECULARA
# =========================================================

with tab_comparison:

    st.header("Comparația dintre două molecule")

    st.write(
        "Introdu două secvențe SMILES pentru a compara "
        "descriptorii moleculari și amprentele Morgan."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        smiles_a = st.text_input(
            "SMILES pentru molecula A:",
            value=DEFAULT_SMILES,
            key="comparison_a"
        )

    with col_b:
        smiles_b = st.text_input(
            "SMILES pentru molecula B:",
            value="CNCCOCC(=O)Nc1ccc(cc1)Br",
            key="comparison_b"
        )

    if st.button("Compară moleculele", type="primary"):

        mol_a = parse_molecule(smiles_a)
        mol_b = parse_molecule(smiles_b)

        if mol_a is None or mol_b is None:
            st.error(
                "Una sau ambele secvențe SMILES nu sunt valide."
            )

        else:
            st.success("Ambele molecule au fost procesate.")

            image_a, image_b = st.columns(2)

            with image_a:
                st.subheader("Molecula A")
                st.image(
                    molecule_image(mol_a, size=(400, 300)),
                    use_container_width=True
                )
                st.code(Chem.MolToSmiles(mol_a))

            with image_b:
                st.subheader("Molecula B")
                st.image(
                    molecule_image(mol_b, size=(400, 300)),
                    use_container_width=True
                )
                st.code(Chem.MolToSmiles(mol_b))

            st.markdown("---")

            # SIMILARITATE TANIMOTO
            fingerprint_a = create_fingerprint(mol_a)
            fingerprint_b = create_fingerprint(mol_b)

            similarity = DataStructs.TanimotoSimilarity(
                fingerprint_a,
                fingerprint_b
            )

            st.subheader("Similaritatea structurală")

            st.metric(
                label="Similaritate Tanimoto",
                value=f"{similarity * 100:.2f}%"
            )

            st.progress(float(similarity))

            st.caption(
                "Valoarea Tanimoto compară amprentele moleculare Morgan "
                "cu raza 2 și 2048 de biți. Nu reprezintă o măsură "
                "directă a similarității activității biologice."
            )

            st.markdown("---")

            # COMPARATIE DESCRIPTORI
            st.subheader("Comparația descriptorilor")

            desc_a = calculate_descriptors(mol_a)
            desc_b = calculate_descriptors(mol_b)

            comparison_data = []

            for descriptor in desc_a:
                comparison_data.append(
                    {
                        "Descriptor": descriptor,
                        "Molecula A": desc_a[descriptor],
                        "Molecula B": desc_b[descriptor]
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            csv_comparison = comparison_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Descarcă comparația în format CSV",
                data=csv_comparison,
                file_name="comparatie_moleculara.csv",
                mime="text/csv"
            )


# =========================================================
# TAB 3 - METODOLOGIE
# =========================================================

with tab_methodology:

    st.header("Metodologia proiectului")

    st.subheader("1. Scopul aplicației")

    st.write(
        """
        Molecular Explorer este o aplicație experimentală de
        chimioinformatică, concepută pentru explorarea structurilor
        moleculare și calcularea unor proprietăți fizico-chimice.
        """
    )

    st.subheader("2. Reprezentarea SMILES")

    st.write(
        """
        SMILES (Simplified Molecular Input Line Entry System) este
        o notație textuală utilizată pentru reprezentarea structurilor
        chimice. Aplicația utilizează biblioteca RDKit pentru
        interpretarea și procesarea acestor secvențe.
        """
    )

    st.subheader("3. Descriptorii moleculari")

    st.write(
        """
        Descriptorii moleculari sunt valori calculate pornind de la
        structura unei molecule. În aplicație sunt utilizați:
        """
    )

    st.markdown(
        """
        - Formula moleculară;
        - Masa moleculară;
        - LogP, descriptor asociat lipofilicității;
        - TPSA, suprafața polară topologică;
        - Numărul donorilor și acceptorilor de legături de hidrogen;
        - Numărul legăturilor rotative;
        - Numărul atomilor grei și al inelelor.
        """
    )

    st.subheader("4. Grupele funcționale")

    st.write(
        """
        Grupele funcționale sunt identificate prin intermediul
        unor tipare SMARTS. Acestea permit căutarea unor fragmente
        structurale specifice în moleculă.
        """
    )

    st.subheader("5. Scheletul Bemis–Murcko")

    st.write(
        """
        Scheletul Bemis–Murcko reprezintă o modalitate de descriere
        a nucleului structural al unei molecule, prin păstrarea
        sistemelor ciclice și a legăturilor dintre acestea.
        """
    )

    st.subheader("6. Amprentele Morgan și similaritatea Tanimoto")

    st.write(
        """
        Amprentele Morgan codifică informații despre mediile
        atomice locale. Similaritatea Tanimoto este calculată
        folosind formula:
        """
    )

    st.latex(
        r"T(A,B)=\\frac{|A\\cap B|}{|A\\cup B|}"
    )

    st.write(
        """
        În această formulă, A și B reprezintă mulțimile de biți
        activați din cele două amprente moleculare.
        """
    )

    st.subheader("7. Limitări")

    st.warning(
        """
        Rezultatele sunt obținute prin calcule informatice și
        nu reprezintă măsurători experimentale. Similaritatea
        structurală nu demonstrează automat o activitate biologică
        asemănătoare. Identificarea grupelor funcționale depinde
        de tiparele SMARTS definite în program.
        """
    )

    st.subheader("8. Instrumente software")

    st.markdown(
        """
        - **Python** – limbajul de programare;
        - **Streamlit** – interfața web;
        - **RDKit** – procesarea structurilor moleculare;
        - **Pandas** – organizarea și exportul datelor.
        """
    )

    st.subheader("9. Bibliografie orientativă")

    st.markdown(
        """
        1. RDKit Documentation – https://www.rdkit.org/docs/
        2. Streamlit Documentation – https://docs.streamlit.io/
        3. Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs.
           5. Molecular Frameworks. Journal of Medicinal Chemistry, 1996.
        4. Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints.
           Journal of Chemical Information and Modeling, 2010.
        """
    )


# ---------------------------------------------------------
# SUBSOL
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    **COLEGIUL TEHNIC „ANA ASLAN” CLUJ-NAPOCA**  
    Județul Cluj

    Realizat în scop experimental de eleva **Apahidean Maria Giulia**  
    Sub îndrumarea profesoarei coordonatoare **Chicinaș Raluca**  
    An școlar **2026–2027**
    """
)

import pandas as pd
import streamlit as st
from io import StringIO
from google.cloud import storage
from utils.config import AppConfig

config = AppConfig()

def blob_as_csv(bucket, path, sep=r"\s+", header="infer"):
    df = pd.read_csv(bucket+path, sep=sep, header=header)
    return df

def blob_as_html(bucket, path):
    blob_str = str(bucket+path, "utf-8")  # Convert bytes to string
    return blob_str 

def get_gcloud_bucket():
    return config.FRONTEND_BUCKET_NAME

def get_master_key():
    release_choice = st.session_state["release_choice"]
    master_key_path = f"/release_keys/nba_app_key.csv"
    master_key = blob_as_csv(config.FRONTEND_BUCKET_NAME, master_key_path, sep=",")
    latest_rel = max(master_key.release)
    if release_choice == latest_rel:
        return master_key
    else:
        return master_key[master_key.release == release_choice]

def filter_by_cohort(master_key):
    master_key = cohort_select(master_key)
    master_key = master_key[master_key["prune_reason"].isnull()]

    return master_key

def filter_by_ancestry(master_key):
    meta_ancestry_select()
    meta_ancestry_choice = st.session_state["meta_ancestry_choice"]
    if meta_ancestry_choice != "All":
        master_key = master_key[master_key["label"] == meta_ancestry_choice]
    return master_key

def update_sex_labels(master_key):
    master_key["sex"].replace(config.SEX_MAP, inplace=True)
    return master_key

def config_page(title):
    if "gp2_bg" in st.session_state:
        st.set_page_config(
            page_title=title,
            page_icon=st.session_state.gp2_bg,
            layout="wide",
        )
    else:
        gp2_bg = config.FRONTEND_BUCKET_NAME+"/gp2_2.jpg" 
        st.session_state["gp2_bg"] = gp2_bg
        st.set_page_config(
            page_title=title,
            page_icon=gp2_bg,
            layout="wide"
        )

def place_logos():
    sidebar1, sidebar2 = st.sidebar.columns(2)
    if ("card_removebg" in st.session_state) and ("redlat" in st.session_state):
        sidebar1.image(st.session_state.card_removebg, use_container_width=True)
        sidebar2.image(st.session_state.gp2_removebg, use_container_width=True)
        st.sidebar.image(st.session_state.redlat, use_container_width=True)
    else:
        card_removebg = config.FRONTEND_BUCKET_NAME+"/card-removebg.png"
        gp2_removebg = config.FRONTEND_BUCKET_NAME+"/gp2_2-removebg.png"

        st.session_state["card_removebg"] = card_removebg
        st.session_state["gp2_removebg"] = gp2_removebg
        # st.session_state["redlat"] = redlat
        sidebar1.image(card_removebg, use_container_width=True)
        sidebar2.image(gp2_removebg, use_container_width=True)
        # st.sidebar.image(redlat, use_container_width=True)

def release_callback():
    st.session_state["old_release_choice"] = st.session_state["release_choice"]
    st.session_state["release_choice"] = st.session_state["new_release_choice"]

def release_select():
    st.sidebar.markdown("### **Choose a release!**")
    release_options = [9] # can replace with master key reference

    if "release_choice" not in st.session_state:
        st.session_state["release_choice"] = release_options[0]
    if "old_release_choice" not in st.session_state:
        st.session_state["old_release_choice"] = ""

    st.session_state["release_choice"] = st.sidebar.selectbox(
        label="Release Selection",
        label_visibility="collapsed",
        options=release_options,
        index=release_options.index(st.session_state["release_choice"]),
        key="new_release_choice",
        on_change=release_callback
    )

def cohort_callback():
    """
    Update session state upon changing the cohort selection.
    """
    st.session_state["old_cohort_choice"] = st.session_state["cohort_choice"]
    st.session_state["cohort_choice"] = st.session_state["new_cohort_choice"]

def cohort_select(master_key):
    """
    Sidebar widget for selecting the cohort.
    """
    st.sidebar.markdown("### **Choose a cohort!**", unsafe_allow_html=True)

    release_value = st.session_state["release_choice"]
    options = [f"GP2 Release {release_value} FULL"] + list(master_key["study"].unique())
    full_release_options = [f"GP2 Release {i} FULL" for i in range(1, 9)]

    if "cohort_choice" not in st.session_state:
        st.session_state["cohort_choice"] = options[0]

    if st.session_state["cohort_choice"] not in options:
        if st.session_state["cohort_choice"] not in full_release_options:
            st.error(
                f"Cohort: {st.session_state['cohort_choice']} not available for "
                f"GP2 Release {release_value}. Displaying GP2 Release "
                f"{release_value} FULL instead!"
            )
        st.session_state["cohort_choice"] = options[0]

    if "old_cohort_choice" not in st.session_state:
        st.session_state["old_cohort_choice"] = ""

    st.session_state["cohort_choice"] = st.sidebar.selectbox(
        label="Cohort Selection",
        label_visibility="collapsed",
        options=options,
        index=options.index(st.session_state["cohort_choice"]),
        key="new_cohort_choice",
        on_change=cohort_callback
    )

    if st.session_state["cohort_choice"] == f"GP2 Release {release_value} FULL":
        st.session_state["master_key"] = master_key
    else:
        master_key_cohort = master_key[master_key["study"] == st.session_state["cohort_choice"]]
        st.session_state["master_key"] = master_key_cohort

    # pruned_counts = st.session_state.master_key["pruned"].value_counts()
    # pruned_samples = pruned_counts[1] if 1 in pruned_counts else 0
    pruned_samples = len(st.session_state.master_key[st.session_state.master_key["prune_reason"].notnull()])
    total_count = st.session_state["master_key"].shape[0]

    st.sidebar.metric(" ", st.session_state["cohort_choice"])
    st.sidebar.metric("Number of Samples in Dataset:", f"{total_count:,}")
    st.sidebar.metric("Number of Samples After Pruning:", f"{(total_count - pruned_samples):,}")

    st.sidebar.markdown("---")
    place_logos()
    return st.session_state["master_key"]

def meta_ancestry_callback():
    """
    Update session state upon changing the meta ancestry selection.
    """
    st.session_state["old_meta_ancestry_choice"] = st.session_state["meta_ancestry_choice"]
    st.session_state["meta_ancestry_choice"] = st.session_state["new_meta_ancestry_choice"]

def meta_ancestry_select():
    """
    Widget for selecting meta ancestry to filter the master key.
    """
    st.markdown("#### **Choose an ancestry:**")
    master_key = st.session_state["master_key"]
    meta_ancestry_options = ["All"] + list(master_key["label"].dropna().unique())
    if "meta_ancestry_choice" not in st.session_state:
        st.session_state["meta_ancestry_choice"] = meta_ancestry_options[0]

    if st.session_state["meta_ancestry_choice"] not in meta_ancestry_options:
        st.error(
            f"No samples with {st.session_state['meta_ancestry_choice']} ancestry in "
            f"{st.session_state['cohort_choice']}. Displaying all ancestries instead!"
        )
        st.session_state["meta_ancestry_choice"] = meta_ancestry_options[0]

    if "old_meta_ancestry_choice" not in st.session_state:
        st.session_state["old_meta_ancestry_choice"] = ""

    st.session_state["meta_ancestry_choice"] = st.selectbox(
        label="Ancestry Selection",
        label_visibility="collapsed",
        options=meta_ancestry_options,
        index=meta_ancestry_options.index(st.session_state["meta_ancestry_choice"]),
        key="new_meta_ancestry_choice",
        on_change=meta_ancestry_callback
    )

def admix_ancestry_callback():
    """
    Update session state upon changing the admixture ancestry selection.
    """
    st.session_state["old_admix_ancestry_choice"] = st.session_state["admix_ancestry_choice"]
    st.session_state["admix_ancestry_choice"] = st.session_state["new_admix_ancestry_choice"]

def admix_ancestry_select(proj_labels):
    """
    Widget for selecting admixture ancestry to filter the master key.
    """
    st.markdown("### **Choose an ancestry!**")

    admix_ancestry_options = ["All"] + list(proj_labels["Predicted Ancestry"].dropna().unique())

    if "admix_ancestry_choice" not in st.session_state:
        st.session_state["admix_ancestry_choice"] = admix_ancestry_options[0]
    if "old_admix_ancestry_choice" not in st.session_state:
        st.session_state["old_admix_ancestry_choice"] = ""

    st.session_state["admix_ancestry_choice"] = st.selectbox(
        label="Ancestry Selection",
        label_visibility="collapsed",
        options=admix_ancestry_options,
        index=admix_ancestry_options.index(st.session_state["admix_ancestry_choice"]),
        key="new_admix_ancestry_choice",
        on_change=admix_ancestry_callback
    )

def chr_callback():
    """
    Update session state upon changing the chromosome selection.
    """
    st.session_state["old_chr_choice"] = st.session_state["chr_choice"]
    st.session_state["chr_choice"] = st.session_state["new_chr_choice"]

def ancestry_callback():
    """
    Update session state upon changing the ancestry selection.
    """
    st.session_state["old_ancestry_choice"] = st.session_state["ancestry_choice"]
    st.session_state["ancestry_choice"] = st.session_state["new_ancestry_choice"]

def chr_ancestry_select():
    """
    Sidebar widgets for selecting chromosome and ancestry.
    """
    st.sidebar.markdown("### **Choose a chromosome!**", unsafe_allow_html=True)
    chr_options = list(range(1, 23))

    if "chr_choice" not in st.session_state:
        st.session_state["chr_choice"] = chr_options[0]
    if "old_chr_choice" not in st.session_state:
        st.session_state["old_chr_choice"] = ""

    st.session_state["chr_choice"] = st.sidebar.selectbox(
        label="Chromosome Selection",
        label_visibility="collapsed",
        options=chr_options,
        index=chr_options.index(st.session_state["chr_choice"]),
        key="new_chr_choice",
        on_change=chr_callback
    )

    st.sidebar.markdown("### **Choose an Ancestry!**", unsafe_allow_html=True)
    ancestry_options = config.ANCESTRY_OPTIONS

    if "ancestry_choice" not in st.session_state:
        st.session_state["ancestry_choice"] = ancestry_options[0]
    if "old_ancestry_choice" not in st.session_state:
        st.session_state["old_ancestry_choice"] = ""

    st.session_state["ancestry_choice"] = st.sidebar.selectbox(
        label="Ancestry Selection",
        label_visibility="collapsed",
        options=ancestry_options,
        index=ancestry_options.index(st.session_state["ancestry_choice"]),
        key="new_ancestry_choice",
        on_change=ancestry_callback
    )

    st.sidebar.markdown("---")
    place_logos()

def rv_cohort_callback():
    """
    Update session state upon changing the rare-variant cohort selection.
    """
    st.session_state["old_rv_cohort_choice"] = st.session_state["rv_cohort_choice"]
    st.session_state["rv_cohort_choice"] = st.session_state["new_rv_cohort_choice"]

def method_callback():
    """
    Update session state upon changing the method selection.
    """
    st.session_state["old_method_choice"] = st.session_state["method_choice"]
    st.session_state["method_choice"] = st.session_state["new_method_choice"]

def rv_gene_callback():
    """
    Update session state upon changing the gene selection.
    """
    st.session_state["old_rv_gene_choice"] = st.session_state["rv_gene_choice"]
    st.session_state["rv_gene_choice"] = st.session_state["new_rv_gene_choice"]

def rv_select(rv_data):
    """
    Sidebar widgets for selecting rare-variant parameters:
    cohort, methods, and gene.
    """
    st.sidebar.markdown("### **Choose a cohort!**", unsafe_allow_html=True)
    rv_cohort_options = list(rv_data["Study code"].unique())

    if "rv_cohort_choice" not in st.session_state:
        st.session_state["rv_cohort_choice"] = None
    if "old_rv_cohort_choice" not in st.session_state:
        st.session_state["old_rv_cohort_choice"] = ""

    st.session_state["rv_cohort_choice"] = st.sidebar.multiselect(
        label="Cohort Selection",
        label_visibility="collapsed",
        options=rv_cohort_options,
        default=st.session_state["rv_cohort_choice"],
        key="new_rv_cohort_choice",
        on_change=rv_cohort_callback
    )

    st.sidebar.markdown("### **Choose a discovery method!**", unsafe_allow_html=True)
    method_options = list(rv_data["Methods"].unique())

    if "method_choice" not in st.session_state:
        st.session_state["method_choice"] = None
    if "old_method_choice" not in st.session_state:
        st.session_state["old_method_choice"] = ""

    st.session_state["method_choice"] = st.sidebar.multiselect(
        label="Method Selection",
        label_visibility="collapsed",
        options=method_options,
        default=st.session_state["method_choice"],
        key="new_method_choice",
        on_change=method_callback
    )

    st.sidebar.markdown("### **Choose a gene!**", unsafe_allow_html=True)
    rv_gene_options = list(rv_data["Gene"].unique())

    if "rv_gene_choice" not in st.session_state:
        st.session_state["rv_gene_choice"] = None
    if "old_rv_gene_choice" not in st.session_state:
        st.session_state["old_rv_gene_choice"] = ""

    st.session_state["rv_gene_choice"] = st.sidebar.multiselect(
        label="Gene Selection",
        label_visibility="collapsed",
        options=rv_gene_options,
        default=st.session_state["rv_gene_choice"],
        key="new_rv_gene_choice",
        on_change=rv_gene_callback
    )

    st.sidebar.markdown("---")
    place_logos()

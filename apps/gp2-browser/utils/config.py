from typing import Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict
# bucket_flag = "gcs"
class AppConfig(BaseSettings):
    """
    Centralized configuration using Pydantic for validation and defaults.
    """
    # GCP_PROJECT: str = "gp2-code-test-env" #"gp2-release-terra"
    FRONTEND_BUCKET_NAME: str = "data/gt_app_utils"

    SEX_MAP: Dict[int, str] = {
        1: "Male",
        2: "Female",
        0: "Unknown",
    }

    PRUNE_MAP: Dict[str, str] = {
        "callrate": "Callrate Prune",
        "duplicated": "Duplicated Prune",
        "sex": "Sex Prune",
        "het": "Heterozygosity Prune",
        "missing_idat": "Missing IDAT File",
        "missing_bed": "Missing BED File",
        "clinical_inconsistency": "Inconsistency with Clinical Manifests"
    }

    ANCESTRY_OPTIONS: List[str] = [
        "AAC","AFR","AJ","AMR","CAH","CAS","EAS","EUR","FIN","MDE","SAS"
    ]

    ANCESTRY_COLOR_MAP: Dict[str, str] = {
    'AFR': "#88CCEE",
    'SAS': "#CC6677",
    'EAS': "#DDCC77",
    'EUR': "#117733",
    'AMR': "#332288",
    'AJ': "#D55E00",
    'AAC': "#999933",
    'CAS': "#882255",
    'MDE': "#661100",
    'FIN': "#F0E442",
    'CAH': "#40B0A6",
    'Predicted': "#ababab",
    'Other Ancestries': '#D3D3D3'
    }

    DESCRIPTIONS: Dict[str, str] = {
        'qc': 'Genotypes are pruned for call rate with maximum sample genotype missingness of 0.02 (--mind 0.02). Samples which pass\
                call rate pruning are then pruned for discordant sex where samples with 0.25 <= sex F <= 0.75 are pruned. Sex F < 0.25\
                are female and Sex F > 0.75 are male. Samples that pass sex pruning are then differentiated by ancestry (refer to\
                ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness using KING, where a  cutoff of 0.0884 \
                was used to determine second degree relatedness and 0.354 is used to determine duplicates. For purposes of imputation,\
                related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity where F <= -0.25 of\
                F>= 0.25.',
        'variant': 'Variants are pruned for missingness by case-control where P<=1e-4 to detect platform/batch differences in case-control status.\
                Next, variants are pruned for missingness by haplotype for flanking variants where P<=1e-4. Lastly, controls are filtered for HWE \
                at a threshold of 1e-4. Please note that for each release, variant pruning is performed in an ancestry-specific manner, and thus \
                the numbers in the bar chart below will not change based on cohort selection within the same release.',
        'pca1':'Select an Ancestry Category below to display only the Predicted samples within that label.',
        'pca2':'All Predicted samples and their respective labels are listed below. Click on the table and use ⌘ Cmd + F or Ctrl + F to search for specific samples.',
        'admixture':'Results of running ADMIXTURE on the reference panel with K=10. Use the selector to subset the admixture table by ancestry. Clicking on a column once or twice will display the table in ascending or descending order, respectively, in terms of that column.',
        'ancestry_methods': """
            ## _Ancestry_
            ### _Reference Panel_
            The reference panel is composed of 4008 samples from 1000 Genomes Project, Human Genome Diversity Project (HGDP),
            and an Ashkenazi Jewish reference panel (Gene Expression Omnibus (GEO) database, accession no. GSE23636) with the
            following ancestral makeup:

            - African (AFR): 819
            - African Admixed and Caribbean (AAC): 74
            - Ashkenazi Jewish (AJ): 471
            - Central Asian (CAS): 183
            - East Asian (EAS): 585
            - European (EUR): 534
            - Finnish (FIN): 99
            - Latino/American Admixed (AMR): 490
            - Middle Eastern (MDE): 152
            - South Asian (SAS): 601

            Samples were chosen from 1000 Genomes and HGDP to match the specific ancestries present in GP2. The reference panel
            was then pruned for palindrome SNPs (A1A2= AT or TA or GC or CG). SNPs were then pruned for maf 0.05, geno 0.01,
            and hwe 0.0001.

            ### _Preprocessing_
            The genotypes were pruned for geno 0.1. Common variants between the reference panel and the genotypes were extracted
            from both the reference panel and the genotypes. Any missing genotypes were imputed using the mean of that particular
            variant in the reference panel.

            The reference panel samples were split into an 80/20 train/test set and then PCs were fit to and transformed the
            training set using sklearn PCA.fit_transform. The test set was transformed using sklearn PCA.transform and normalized
            to the training set parameters. Genotypes were then transformed in the same way for prediction.

            ### _UMAP + Classifier Training_
            A classifier was then trained using UMAP transformations of the PCs and a linear XGBoost classifier using a 5-fold
            cross-validation. It was scored for balanced accuracy with a gridsearch over specific parameters:

            - “umap__n_neighbors”: [5,20]
            - “umap__n_components”: [15,25]
            - “umap__a”: [0.75, 1.0, 1.5]
            - “umap__b”: [0.25, 0.5, 0.75]
            - “xgboost__lambda”: [0.001, 0.01, 0.1, 1, 10, 100]

            Performance varies from 95-98% balanced accuracy on the test set depending on overlapping genotypes.

            ### _Prediction_
            Scaled PCs for genotypes are transformed using UMAP fitted on the training set and then predicted by the classifier.
            Genotypes are split and output into individual ancestries. Prior to release 5, AAC and AFR labels were combined into a
            single category and ADMIXTURE 1 was run with --supervised to further divide these two categories. From release 5 on,
            the AFR and AAC sample labels in the reference panel are adjusted using a perceptron model, and the predictions
            effectively estimate the results from the ADMIXTURE step.

            ### _Complex Admixture History_
            Certain highly admixed ancestry groups are not well-represented by the constructed reference panel used by GenoTools.
            Due to a lack of publicly available reference samples for these groups, GenoTools employs a method to identify such samples
            and place them in an ancestry group not present in the reference panel, labeled 'Complex Admixture History' (CAH).
            Any sample whose PC distance is closer to the overall PC centroid of the training data than to any reference panel
            ancestry group centroid is labeled as CAH.
            """,
        'snp_metrics':'To arrive at the set of variants for which SNP metrics are available, variants underwent additional pruning \
                after the steps described on the Quality Control page. Within each ancestry, variants were pruned for call rate \
                with a maximum variant genotype missingness of 0.01 (--geno 0.01), a minumum minor allele frequency of 0.01 (--maf 0.01) \
                and HWE at a thresthold of 5e-6. LD pruning was performed to find and prune any pairs of variants with r\u00b2 > 0.02 \
                in a sliding window of 1000 variants with a step size of 10 variants (--indep-pairwise 1000 10 0.02). The SNPs are on build hg38, \
                and this is reflected in the chromosome\:position labels that are available next to the SNP name in the selection box. \
                Please note that SNP Metrics are only available for the most recent GP2 release (GP2 Release 8).'
    }

    HOME_CONTENT: Dict[str, str] = {
        "TITLE": """
    <h1 style='text-align: center; color: #0f557a; font-family: Helvetica;'>
        GP2 Internal Cohort Browser
    </h1>
    """,
        "INTRO": """
    <h5 style='text-align: center;'>
        Interactive tool to visualize quality control and ancestry prediction summary statistics
        across all GP2 cohorts. Please select a page marked with \U0001F9EC in the sidebar to begin.
    </h5>
    """,
        "FULL_DESCRIPTION": """
    <style>
    [data-testid="stMarkdownContainer"] ul {
        padding-left: 40px;
    }
    </style>

    ## _GenoTools Paper Out Now!_
    For a more in-depth description of the methods used to process the GP2 releases, please see the [GenoTools](https://doi.org/10.1093/g3journal/jkae268) paper published in G3

    ## _Quality Control_
    ### _Sample-Level Pruning_
    Genotypes are pruned for call rate with maximum sample genotype missingness of 0.02 (--mind 0.02). Samples which pass
    call rate pruning are then pruned for discordant sex where samples with 0.25 <= sex F <= 0.75 are pruned. Sex F < 0.25
    are female and Sex F > 0.75 are male. Samples that pass sex pruning are then differentiated by ancestry (refer to
    ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness using KING, where a cutoff of
    0.0884 was used to determine second-degree relatedness and 0.354 is used to determine duplicates. For purposes of
    imputation, related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity
    where F <= -0.25 or F >= 0.25.

    ### _Variant-Level Pruning_
    Variants are pruned for missingness by case-control where P <= 1e-4 to detect platform/batch differences in case-control status.
    Next, variants are pruned for missingness by haplotype for flanking variants where P <= 1e-4. Lastly, controls are filtered
    for HWE at a threshold of 1e-4.

    ## _Ancestry_
    ### _Reference Panel_
    The reference panel is composed of 4008 samples from 1000 Genomes Project, Human Genome Diversity Project (HGDP),
    and an Ashkenazi Jewish reference panel (Gene Expression Omnibus (GEO) database, www.ncbi.nlm.nih.gov/geo
    (accession no. GSE23636)) (REF) with the following ancestral makeup:

    - African (AFR): 819
    - African Admixed and Caribbean (AAC): 74
    - Ashkenazi Jewish (AJ): 471
    - Central Asian (CAS): 183
    - East Asian (EAS): 585
    - European (EUR): 534
    - Finnish (FIN): 99
    - Latino/American Admixed (AMR): 490
    - Middle Eastern (MDE): 152
    - South Asian (SAS): 601

    Samples were chosen from 1000 Genomes and HGDP to match the specific ancestries present in GP2. The reference panel was then
    pruned for palindrome SNPs (A1A2 = AT or TA or GC or CG). SNPs were then pruned for maf 0.05, geno 0.01, and hwe 0.0001.

    ### _Preprocessing_
    The genotypes were pruned for geno 0.1. Common variants between the reference panel and the genotypes were extracted
    from both the reference panel and the genotypes. Any missing genotypes were imputed using the mean of that particular
    variant in the reference panel.

    The reference panel samples were split into an 80/20 train/test set and then PCs were fit to and transformed the
    training set using sklearn PCA.fit_transform and mean and standard deviation were extracted for scaling (𝑥/2) * (1 − (𝑥/2))
    where 𝑥 is the mean training set PCs. The test set was then transformed by the PCs in the training set using
    sklearn PCA.transform and then normalized to the mean and standard deviation of the training set. Genotypes were then
    transformed by the same process as the test set for prediction after model training.

    ### _UMAP + Classifier Training_
    A classifier was then trained using UMAP transformations of the PCs and a linear XGBoost classifier using a 5-fold
    cross-validation using an sklearn pipeline and scored for balanced accuracy with a gridsearch over the following parameters:

    - “umap__n_neighbors”: [5, 20]
    - “umap__n_components”: [15, 25]
    - “umap__a”: [0.75, 1.0, 1.5]
    - “umap__b”: [0.25, 0.5, 0.75]
    - “xgboost__lambda”: [0.001, 0.01, 0.1, 1, 10, 100]

    Performance varies from 95-98% balanced accuracy on the test set depending on overlapping genotypes.

    ### _Prediction_
    Scaled PCs for genotypes are transformed using UMAP trained on the training set and then predicted by the classifier.
    Genotypes are split and output into individual ancestries. Prior to release 5, AAC and AFR labels were combined into a
    single category and ADMIXTURE 1 (v1.3.0-https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz)
    was run using the --supervised functionality to further divide these two categories where AFR was assigned if AFR admixture
    was >= 90% and AAC was assigned if AFR admixture was < 90%. From release 5 on, the AFR and AAC sample labels in the reference panel
    are adjusted using a perceptron model, and the predictions based on the updated reference panel labels effectively estimate the
    results from the ADMIXTURE step that was previously used.

    ### _Complex Admixture History_
    Certain highly admixed ancestry groups are not well-represented by the constructed reference panel used by GenoTools.
    Due a lack of publicly available reference samples for highly admixed groups, GenoTools employs a method to identify samples
    of this nature and place them in an ancestry group that is not present in the reference panel, named “Complex Admixture History” (CAH).
    Highly admixed samples of this nature should be analyzed independently. Since there are no reference samples to base the prediction
    of CAH ancestry on, a PC-based approach is used instead. Using the training data, the PC centroid of each reference panel ancestry
    group is calculated, along with the overall PC centroid. For each new sample, the PC distance from each centroid is then calculated.
    Any sample whose PC distance is closer to the overall PC centroid of the training data than to any reference panel ancestry group
    centroid is labeled as CAH.

    **_References_**
    _D.H. Alexander, J. Novembre, and K. Lange. Fast model-based estimation of ancestry in unrelated individuals.
    Genome Research, 19:1655–1664, 2009._
    """,
        "JS_EXPANDER_STYLE": """
    <script>
    var elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
    if (elements.length > 0) {
        elements[0].style.fontSize = 'large';
        elements[0].style.color = '#0f557a';
    }
    </script>
    """
    }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    

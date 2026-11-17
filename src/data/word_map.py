"""Diagnosis and sex labels from notebook 2.1 cell 2."""

WORD_MAP = {
    "Gender": {
        "male": "Male", "female": "Female", "Male": "Male", "Female": "Female",
        "F": "Female", "M": "Male", "0.0": "Female", "1.0": "Male",
    },
    "Diagnosis": {
        "AD": "AD", "normal": "Control", "NPH": "NPH",
        "DLB": "DLB", "MCI": "MCI", "NC": "Control", "VaD": "VaD",
        "NVCID": "NVCID", "VCID": "VCID", "HY": "HY", "Control": "Control",
        "MCI,30": "MCI,30", "Severe": "Severe AD", "Incipient": "Incipient AD",
        "Moderate": "Moderate AD", "ad": "AD", "pd": "PD", "nc": "Control",
        "Alzheimer Disease": "AD", "control": "Control", "Alzheimer's Disease": "AD",
        "Healthy Control": "Control", "alzheimer patient": "AD",
        "early stage (control)": "Early Stage AD", "late stage": "Late Stage AD",
        "Sham": "Control", "Normal Aging": "Control",
        "Mild cognitive impairment due to Alzheimer's disease": "MCI",
        "mild cognitive impairment (MCI)": "MCI", "Normal Young": "Control",
        "Normal Old": "Control", "1.0": "Control", "2.0": "MCI", "3.0": "MCI+",
        "4.0": "AD", "5.0": "AD+", "6.0": "Other", "Health": "Control",
    },
}

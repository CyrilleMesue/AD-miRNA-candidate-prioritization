"""miRBase accession to miRTarBase product name. Notebook 1.1."""

def parse_mirna_dat(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = file.readlines()
    mirna_records = []
    current = {}
    for line in data:
        if line.startswith("FT                   /accession"):
            if current:
                mirna_records.append(current)
                current = {}
            current["FT                   /accession"] = line.split()[1].split('"')[-2]
        elif line.startswith("FT                   /product"):
            current["FT                   /product"] = line.split()[1].rstrip(";").split('"')[-2]
        elif line.startswith("//"):
            if current:
                mirna_records.append(current)
                current = {}
    mapping = {}
    for rec in mirna_records:
        acc = rec.get("FT                   /accession")
        prod = rec.get("FT                   /product")
        if acc and prod:
            mapping[acc] = prod
    return mapping

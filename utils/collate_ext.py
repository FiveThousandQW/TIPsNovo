from ..transformer.dataset import collate_batch

def collate_with_group(batch):
    if not batch:
        return collate_batch(batch)
    if len(batch[0]) == 5:
        groups = [b[4] for b in batch]
        core = [(b[0], b[1], b[2], b[3]) for b in batch]
        spectra, precursors, spectra_mask, peptides, peptides_mask = collate_batch(core)
        return spectra, precursors, spectra_mask, peptides, peptides_mask, groups
    return collate_batch(batch)
import os
import pandas as pd
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

class ColabDockPipeline:
    def __init__(self, working_dir="docking_project"):
        self.working_dir = working_dir
        self.prot_dir = os.path.join(working_dir, "protein_data")
        self.lig_dir = os.path.join(working_dir, "ligand_data")
        os.makedirs(self.prot_dir, exist_ok=True)
        os.makedirs(self.lig_dir, exist_ok=True)
        
    def prepare_protein(self, pdb_id):
        """Downloads a target PDB structure and converts it cleanly to PDBQT format."""
        pdb_path = os.path.join(self.prot_dir, f"{pdb_id}.pdb")
        pdbqt_path = os.path.join(self.prot_dir, f"{pdb_id}_receptor.pdbqt")
        
        # Pull directly from RCSB server
        os.system(f"wget -q -O {pdb_path} https://files.rcsb.org/download/{pdb_id}.pdb")
        
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "pdbqt")
        mol = openbabel.OBMol()
        
        if obConversion.ReadFile(mol, pdb_path):
            mol.DeleteHydrogens()
            mol.AddHydrogens(False, True, 7.4) # Target polar hydrogens for Vina calculations
            obConversion.WriteFile(mol, pdbqt_path)
            print(f"🧬 Target protein {pdb_id} parsed successfully: {pdbqt_path}")
            return pdbqt_path
        else:
            raise ValueError(f"Failed to process or read PDB ID: {pdb_id}")

    def prepare_ligand(self, smiles, ligand_name):
        """Generates 3D coordinates from SMILES and parameterizes partial charges with Meeko."""
        pdbqt_out = os.path.join(self.lig_dir, f"{ligand_name}.pdbqt")
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string, is_valid, error_msg = PDBQTWriterLegacy.write_string(preparator)
        
        if is_valid:
            with open(pdbqt_out, "w") as f:
                f.write(pdbqt_string)
            return pdbqt_out
        else:
            raise ValueError(f"Meeko parameterization failed for {ligand_name}: {error_msg}")

    def run_docking(self, receptor_pdbqt, ligand_pdbqt, center, size, output_name, exhaustiveness=16):
        """Executes native core calculations using the Vina docking engine."""
        output_pose = os.path.join(self.lig_dir, f"{output_name}_poses.pdbqt")
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=center, size=size)
        
        v.dock(exhaustiveness=exhaustiveness, n_poses=9)
        v.write_poses(output_pose, n_poses=9, overwrite=True)
        return v.energies(n_poses=1)[0][0]

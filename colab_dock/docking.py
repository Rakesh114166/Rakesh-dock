import os
import pandas as pd
from vina import Vina

class ColabDockPipeline:
    def __init__(self, working_dir="docking_project"):
        self.working_dir = working_dir
        self.prot_dir = os.path.join(working_dir, "protein_data")
        self.lig_dir = os.path.join(working_dir, "ligand_data")
        os.makedirs(self.prot_dir, exist_ok=True)
        os.makedirs(self.lig_dir, exist_ok=True)
        
    def prepare_protein(self, pdb_id):
        """Downloads a target PDB structure and converts it cleanly to PDBQT via command-line OpenBabel."""
        pdb_path = os.path.join(self.prot_dir, f"{pdb_id}.pdb")
        pdbqt_path = os.path.join(self.prot_dir, f"{pdb_id}_receptor.pdbqt")
        
        os.system(f"wget -q -O {pdb_path} https://files.rcsb.org/download/{pdb_id}.pdb")
        
        cmd = f"obabel {pdb_path} -O {pdbqt_path} -h -xr -d --quiet"
        exit_code = os.system(cmd)
        
        if exit_code == 0 and os.path.exists(pdbqt_path):
            return pdbqt_path
        else:
            raise ValueError(f"Failed to process or convert PDB ID via OpenBabel: {pdb_id}")

    def prepare_ligand(self, smiles, ligand_name):
        """Generates 3D coordinates directly from SMILES lines and parameters bonds via OpenBabel."""
        pdbqt_out = os.path.join(self.lig_dir, f"{ligand_name}.pdbqt")
        
        cmd = f'obabel -:"{smiles}" -O {pdbqt_out} --gen3d -h -p 7.4 --quiet'
        exit_code = os.system(cmd)
        
        if exit_code == 0 and os.path.exists(pdbqt_out):
            return pdbqt_out
        else:
            raise ValueError(f"OpenBabel conversion failed for ligand: {ligand_name}")

    def run_docking(self, receptor_pdbqt, ligand_pdbqt, center, size, output_name, exhaustiveness=16):
        """Executes native screening calculations using the AutoDock Vina engine."""
        output_pose = os.path.join(self.lig_dir, f"{output_name}_poses.pdbqt")
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=9)
        v.write_poses(output_pose, n_poses=9, overwrite=True)
        return v.energies(n_poses=1)[0][0]

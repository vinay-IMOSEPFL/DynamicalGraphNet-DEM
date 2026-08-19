# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class DemDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed Discrete Element Method (DEM) graphs.

    This dataset supports two loading modes:
    1. Case-specific loading: Deep searches the root directory for a specific case
       (e.g., 'case_07') and loads its graph sequence.
    2. Split-based loading: Loads all cases contained within a specified split
       directory (e.g., 'train', 'val', 'test').
    """
    def __init__(self, root: str, split: str = "train", case_name: str = None):
        """
        Initializes the dataset by loading graph data from disk into memory.

        Args:
            root (str): The base directory containing the dataset.
            split (str): The sub-directory split to load (e.g., 'heterogeneous/gravity/training').
                         Ignored if case_name is provided.
            case_name (str, optional): A specific case folder to locate and load.
                                       If provided, the class searches the entire root tree.
        """
        self.root = root
        self.graph_list = []

        if case_name is not None:
            # Mode 1: Deep search for a specific case across all nested folders
            found_path = None

            # Walk the directory tree to locate the target case folder
            for dirpath, dirnames, filenames in os.walk(root):
                potential_path = os.path.join(dirpath, case_name, "graph_list.pt")
                if os.path.isfile(potential_path):
                    found_path = potential_path
                    self.split_dir = os.path.join(dirpath, case_name)
                    break

            if found_path is None:
                raise ValueError(f"Case '{case_name}' with a 'graph_list.pt' was not found anywhere under {root}.")

            # weights_only=False is required from PyTorch 2.6 onwards: these files
            # hold pickled PyTorch Geometric `Data` objects, not plain state dicts.
            graphs = torch.load(found_path, weights_only=False)
            if not isinstance(graphs, (list, tuple)):
                raise ValueError(f"`graph_list.pt` in {case_name} must be a list of Data, got {type(graphs)}")

            self.graph_list.extend(graphs)

        else:
            # Mode 2: Load all cases within a specific split directory
            self.split_dir = os.path.join(root, split)
            if not os.path.isdir(self.split_dir):
                raise ValueError(f"No such split directory: {self.split_dir}")

            # Identify all valid case subdirectories within the split
            cases = sorted(
                d for d in os.listdir(self.split_dir)
                if os.path.isdir(os.path.join(self.split_dir, d))
            )

            if not cases:
                raise ValueError(f"No case subfolders found under {self.split_dir}")

            # Iterate through each case and append its graphs to the master list
            for case in cases:
                path = os.path.join(self.split_dir, case, "graph_list.pt")
                if not os.path.isfile(path):
                    print(f"Warning: Skipping '{case}' as no 'graph_list.pt' was found.")
                    continue

                graphs = torch.load(path, weights_only=False)
                if not isinstance(graphs, (list, tuple)):
                    raise ValueError(f"`graph_list.pt` in {case} must be a list of Data, got {type(graphs)}")

                self.graph_list.extend(graphs)

        if len(self.graph_list) == 0:
            raise ValueError("No graphs loaded from the specified path(s). Check your directories.")

    def __len__(self):
        """Returns the total number of graphs in the loaded dataset."""
        return len(self.graph_list)

    def __getitem__(self, idx: int) -> Data:
        """Retrieves a single graph at the specified index."""
        return self.graph_list[idx]

    def get_stats(self, device=None):
        """
        Computes global dataset statistics required for feature normalization.

        This batches the entire dataset into a single PyTorch Geometric graph
        to efficiently compute the minimum, maximum, mean, and standard deviation
        of the input features and target labels.

        Args:
            device (torch.device, optional): The device on which to perform the computations.

        Returns:
            dict: A dictionary containing the computed statistics for node velocities,
                  angular velocities, edge attributes, and target displacements/changes.
        """
        # Batch the entire dataset into memory to vectorize the reduction operations
        loader = DataLoader(self, batch_size=len(self), shuffle=False)
        stat_graph = next(iter(loader))

        if device is not None:
            stat_graph = stat_graph.to(device)

        # Compute vector magnitudes (L2 norms) for input features
        all_vel = stat_graph.vel.norm(dim=1, keepdim=True)
        all_angvel = stat_graph.ang_vel.norm(dim=1, keepdim=True)
        all_edge_feat = stat_graph.edge_attr.norm(dim=1, keepdim=True)

        # Extract target labels (velocity change, displacement, angular velocity change)
        all_dv = stat_graph.y_dv
        all_dx = stat_graph.y_dx
        all_dw = stat_graph.y_dw

        # Aggregate statistics
        stats = {
            "node_vel_min": all_vel.min(),
            "node_vel_max": all_vel.max(),
            "node_angvel_min": all_angvel.min(),
            "node_angvel_max": all_angvel.max(),
            "edge_feat_min": all_edge_feat.min(),
            "edge_feat_max": all_edge_feat.max(),

            # Ground truth targets use standardization (mean and std)
            "dv_mean": all_dv.mean(dim=0),
            "dv_std": all_dv.std(dim=0),
            "dw_mean": all_dw.mean(dim=0),
            "dw_std": all_dw.std(dim=0),
            "dx_mean": all_dx.mean(dim=0),
            "dx_std": all_dx.std(dim=0),
        }

        return stats
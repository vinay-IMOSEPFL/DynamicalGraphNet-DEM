# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (EPFL),
# Switzerland, 2025.
# Authors: Vinay Sharma and Olga Fink, Laboratory of Intelligent Maintenance and
# Operations Systems (IMOS).
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class DemDataset(Dataset):
    """
    A Dataset that reads `graph_list.pt` files.
    - If `case_name` is provided, it automatically searches all directories to find and load just that case.
    - If `case_name` is None, it loads all cases within the specified `split` directory.
    """
    def __init__(self, root: str, split: str = "train", case_name: str = None):
        self.root = root
        self.graph_list = []

        if case_name is not None:
            # Mode 1: Deep search for a specific case across all nested folders
            found_path = None

            # os.walk completely maps out the directory tree, no matter how deep
            for dirpath, dirnames, filenames in os.walk(root):
                # Check if appending the case_name to the current directory leads to our file
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
            # Mode 2: Original behavior, load all cases in a specific split
            self.split_dir = os.path.join(root, split)
            if not os.path.isdir(self.split_dir):
                raise ValueError(f"No such split directory: {self.split_dir}")

            cases = sorted(
                d for d in os.listdir(self.split_dir)
                if os.path.isdir(os.path.join(self.split_dir, d))
            )
            if not cases:
                raise ValueError(f"No case subfolders found under {self.split_dir}")

            for case in cases:
                path = os.path.join(self.split_dir, case, "graph_list.pt")
                if not os.path.isfile(path):
                    # Warning instead of crashing, just in case there are empty leftover folders
                    print(f"Warning: Skipping '{case}' as no 'graph_list.pt' was found.")
                    continue

                graphs = torch.load(path, weights_only=False)
                if not isinstance(graphs, (list, tuple)):
                    raise ValueError(f"`graph_list.pt` in {case} must be a list of Data, got {type(graphs)}")
                self.graph_list.extend(graphs)

        if len(self.graph_list) == 0:
            raise ValueError(f"No graphs loaded from the specified path(s). Check your directories.")

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx: int) -> Data:
        return self.graph_list[idx]

    def get_stats(self, device=None):
        """
        Batch the entire dataset into one big graph and compute mean & std values for scaling.
        """
        loader = DataLoader(self, batch_size=len(self), shuffle=False)
        stat_graph = next(iter(loader))
        if device is not None:
            stat_graph = stat_graph.to(device)

        all_vel = stat_graph.vel.norm(dim=1, keepdim=True)
        all_angvel = stat_graph.ang_vel.norm(dim=1, keepdim=True)
        all_edge_feat = stat_graph.edge_attr.norm(dim=1, keepdim=True)

        all_dv = stat_graph.y_dv
        all_dx = stat_graph.y_dx
        all_dw = stat_graph.y_dw

        stats = {
            "node_vel_min": all_vel.min(),
            "node_vel_max":  all_vel.max(),
            "node_angvel_min": all_angvel.min(),
            "node_angvel_max":  all_angvel.max(),
            "edge_feat_min": all_edge_feat.min(),
            "edge_feat_max":  all_edge_feat.max(),
            "dv_mean": all_dv.mean(dim=0),
            "dv_std":  all_dv.std(dim=0),
            "dw_mean": all_dw.mean(dim=0),
            "dw_std":  all_dw.std(dim=0),
            "dx_mean": all_dx.mean(dim=0),
            "dx_std":  all_dx.std(dim=0),
        }
        return stats
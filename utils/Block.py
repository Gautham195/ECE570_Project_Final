
#from utils.gnn_encoder import GNNEncoder
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info


class COMetaModel(pl.LightningModule):
  def __init__(self,
               param_args,
               node_feature_only=False):
    super(COMetaModel, self).__init__()
    self.args = param_args
    self.type_diff = self.args.type_diff
    self.sched_diff = self.args.sched_diff
    self.steps_diff = self.args.steps_diff
    self.sparse = self.args.neighbor_count > 0 or node_feature_only

    if self.type_diff == 'gaussian':
      out_channels = 1
      self.diffusion = GaussianDiffusion(
          T=self.steps_diff, schedule=self.sched_diff)
    elif self.type_diff == 'categorical':
      out_channels = 2
      self.diffusion = CategoricalDiffusion(
          T=self.steps_diff, schedule=self.sched_diff)
    else:
      raise ValueError(f"Unknown diffusion type {self.type_diff}")
    #num_layers, h_dim, output_channels=1, aggregate_method="sum", normalization_method="layer",
    #             affine_flag=True, stats_flag=False, gated_flag=True,
    #             sparse_flag=False, checkpoint_flag=False, node_only=False
    self.model = GNNEncoder(
        num_layers=self.args.n_layers,
        h_dim=self.args.h_dim,
        out_channels=out_channels,
        aggregate_method=self.args.summation,
        sparse_flag=self.sparse,
        checkpoint_flag=self.args.use_activation_checkpoint,
        node_only=node_feature_only,
    )
    self.num_training_steps_cached = None

  def test_epoch_end(self, outputs):
    unmerged_metrics = {}
    for metrics in outputs:
      for k, v in metrics.items():
        if k not in unmerged_metrics:
          unmerged_metrics[k] = []
        unmerged_metrics[k].append(v)

    merged_metrics = {}
    for k, v in unmerged_metrics.items():
      merged_metrics[k] = float(np.mean(v))
    self.logger.log_metrics(merged_metrics, step=self.global_step)

  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }

  def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
    """Sample from the categorical posterior for a given time step.
       See https://arxiv.org/pdf/2107.03006.pdf for details.
    """
    diffusion = self.diffusion

    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    # Thanks to Daniyar and Shengyu, who found the "target_t == 0" branch is not needed :)
    # if target_t > 0:
    Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
    Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
    # else:
    #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
    Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
    Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

    xt = F.one_hot(xt.long(), num_classes=2).float()
    xt = xt.reshape(x0_pred_prob.shape)

    x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
    x_t_target_prob_part_2 = Q_bar_t_target[0]
    x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

    x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

    sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
    x_t_target_prob_part_2_new = Q_bar_t_target[1]
    x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

    x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

    sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

    if target_t > 0:
      xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
    else:
      xt = sum_x_t_target_prob.clamp(min=0)

    if self.sparse:
      xt = xt.reshape(-1)
    return xt

  def gaussian_posterior(self, target_t, t, pred, xt):
    """Sample (or deterministically denoise) from the Gaussian posterior for a given time step.
       See https://arxiv.org/pdf/2010.02502.pdf for details.
    """
    diffusion = self.diffusion
    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    atbar = diffusion.alphabar[t]
    atbar_target = diffusion.alphabar[target_t]

    if self.args.inference_trick is None or t <= 1:
      # Use DDPM posterior
      at = diffusion.alpha[t]
      z = torch.randn_like(xt)
      atbar_prev = diffusion.alphabar[t - 1]
      beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

      xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
      xt_target = xt_target + np.sqrt(beta_tilde).item() * z
    elif self.args.inference_trick == 'ddim':
      xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
      xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
    else:
      raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
    return xt_target

  def duplicate_edge_index(self, edge_index, num_nodes, device):
    """Duplicate the edge index (in sparse graphs) for parallel sampling."""
    edge_index = edge_index.reshape((2, 1, -1))
    edge_index_indent = torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
    edge_index_indent = edge_index_indent * num_nodes
    edge_index = edge_index + edge_index_indent
    edge_index = edge_index.reshape((2, -1))
    return edge_index

  def train_dataloader(self):
    batch_size = self.args.batch_size
    train_dataloader = GraphDataLoader(
        self.train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=self.args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

  def val_dataloader(self):
    batch_size = 1
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader
  

  
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info


class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = TSPDataset(
        file_path=os.path.join(self.args.storage_path, self.args.training_split),
        neighbor_count=self.args.neighbor_count,
    )

    self.test_dataset = TSPDataset(
        file_path=os.path.join(self.args.storage_path, self.args.test_split),
        neighbor_count=self.args.neighbor_count,
    )

    self.validation_dataset = TSPDataset(
        file_path=os.path.join(self.args.storage_path, self.args.validation_split),
        neighbor_count=self.args.neighbor_count,
    )

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    else:
      _, graph_data, point_indicator, edge_indicator, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    if self.sparse:
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
    else:
      t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.type_diff == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.type_diff == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    ns, merge_iterations = 0, 0

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.type_diff == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_steps_diff
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.type_diff == 'gaussian':
          xt = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)

      if self.type_diff == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      # Refine using 2-opt
      solved_tours, ns = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device)
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)

    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": ns,
        f"{split}/merge_iterations": merge_iterations,
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
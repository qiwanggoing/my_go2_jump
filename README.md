# 迁移指南：从 PD 位置控制到残差力矩控制

本文档记录了将 `go2_jump` 任务从一个**模拟 PD 位置控制（Position Control）策略**迁移到一个**残差力矩控制（Residual Torque Control）**策略所做的核心代码修改。

## 1. 核心概念转变

### 原 PD 位置控制 (Before)
在原始版本中，策略网络 (Policy) 输出的是一个目标关节位置（actions）。  
`control.action_scale` 被设置为一个很小的值（例如 0.25），代表弧度。  
`_compute_torques` 函数在软件中实现了一个 PD 控制器，用于计算跟踪这个目标位置所需的力矩。

### 残差力矩控制 (After)
在修改后的版本中，策略网络输出的是一个残差力矩（Residual Torque）。  
PD 控制器仍然存在，但其角色变为一个基础稳定器，始终试图将机器人拉回其默认姿态（`default_dof_pos`）。  
策略网络学习输出一个“额外”的力矩（actions），用于执行跳跃等动态动作。  
`_compute_torques` 函数将“基础 PD 力矩”和“策略残差力矩”相加，得到最终发送给电机的总力矩。

## 2. 关键代码修改

### 步骤 1：修改 `GO2_JUMP_config.py`
- **更改 `control.action_scale` (最关键)**:  
  原代码：  
  ```python
  action_scale = 0.25  # 代表 0.25 弧度
  ```  
  修改后:  
  ```python
  action_scale = 10.0  # 代表 10.0 牛顿·米 (Nm)
  ```

- **更改 `asset.file`**:  
  为了确保仿真器正确处理力矩，将资源文件指向一个明确定义了 effort 驱动模式的 URDF。  
  原代码：  
  ```python
  file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
  ```  
  修改后 (参考 sata 库)：  
  ```python
  file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf'
  ```

### 步骤 2：修改 `go2_jump_env.py`
- **重写 `_compute_torques` 函数**:  
  原代码 (PD 位置跟踪):  
  ```python
  def _compute_torques(self, actions):
      # ...
      p_gains = self.p_gains * self.p_gains_multiplier
      d_gains = self.d_gains * self.d_gains_multiplier
      # 'actions' 是缩放后的目标位置偏移
      torques = p_gains * (actions + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
      return torch.clip(torques, -self.torque_limits, self.torque_limits)
  ```  
  修改后 (残差力矩):  
  ```python
  def _compute_torques(self, actions):
      # ...
      p_gains = self.p_gains * self.p_gains_multiplier
      d_gains = self.d_gains * self.d_gains_multiplier
      # 1. 计算基础 PD 力矩 (用于稳定在默认姿态)
      pd_torques = p_gains * (self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
      # 'actions' 是缩放后的残差力矩
      # 2. 叠加策略网络输出的残差力矩
      torques = pd_torques + actions
      return torch.clip(torques, -self.torque_limits, self.torque_limits)
  ```

- **修改 `_reward_torques` 函数 (逻辑变更)**:  
  原代码:  
  ```python
  def _reward_torques(self):
      # 惩罚最终的总力矩
      return torch.sum(torch.abs(self.torques), dim=1)
  ```  
  修改后 (参考 sata):  
  ```python
  def _reward_torques(self):
      # 惩罚策略网络输出的“残差力矩”（self.actions 是未缩放的策略输出）
      # 这鼓励策略“偷懒”，多依赖基础 PD 控制器
      return torch.sum(torch.square(self.actions), dim=1)
  ```

## 3. 结论
完成以上修改后，策略网络的输出 `actions` 的物理含义从“目标位置”（弧度）彻底转变为“残差力矩”（牛顿·米）。
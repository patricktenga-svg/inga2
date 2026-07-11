# models/rl_agent.py
import torch
import torch.nn as nn
import numpy as np
import os

class Actor(nn.Module):
    """Actor network for PPO"""
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Critic network for PPO"""
    def __init__(self, state_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class RLAgent:
    """Reinforcement Learning agent for dam control"""
    
    def __init__(self, model_path=None, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.is_trained = False
        
        if model_path and os.path.exists(f"{model_path}_actor.pth"):
            self.load(model_path)
    
    def get_action(self, state, deterministic=True):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        
        if deterministic:
            action = action_probs.detach().numpy()[0]
        else:
            noise = np.random.normal(0, 0.05, size=self.action_dim)
            action = np.clip(action_probs.detach().numpy()[0] + noise, 0, 1)
        
        # Denormalize: turbine (0-2200 m³/s), irrigation (0-500 m³/s)
        turbine = action[0] * 2200
        irrigation = action[1] * 500
        
        return {
            'turbine': int(turbine),
            'irrigation': int(irrigation),
            'action_norm': action
        }
    
    def get_recommendation(self, scenario, current_state=None):
        """
        Get operational recommendation based on scenario and current state
        """
        # Rule-based baseline (works without training)
        recommendations = {
            'NORMAL': {'turbine': 1800, 'irrigation': 350, 'action': 'PEAK'},
            'DRY': {'turbine': 1200, 'irrigation': 250, 'action': 'NOMINAL'},
            'VERY DRY': {'turbine': 500, 'irrigation': 150, 'action': 'LOW POWER'}
        }
        
        base_rec = recommendations.get(scenario, recommendations['NORMAL'])
        
        # If trained and state available, refine with RL
        if self.is_trained and current_state is not None:
            rl_action = self.get_action(current_state)
            # Blend RL action with rule-based (70% RL, 30% rule)
            blended_turbine = int(0.7 * rl_action['turbine'] + 0.3 * base_rec['turbine'])
            blended_irrigation = int(0.7 * rl_action['irrigation'] + 0.3 * base_rec['irrigation'])
            
            return {
                'turbine': blended_turbine,
                'irrigation': blended_irrigation,
                'action': base_rec['action'],
                'message': self._get_message(scenario)
            }
        
        return {
            'turbine': base_rec['turbine'],
            'irrigation': base_rec['irrigation'],
            'action': base_rec['action'],
            'message': self._get_message(scenario)
        }
    
    def _get_message(self, scenario):
        messages = {
            'NORMAL': '✅ Conditions idéales – Production maximale recommandée',
            'DRY': '⚠️ Conditions sèches – Adopter une stratégie équilibrée',
            'VERY DRY': '🔴 Sécheresse sévère – Réduire impérativement les lâchers'
        }
        return messages.get(scenario, 'Operation normale recommandée')
    
    def save(self, path_prefix):
        """Save models"""
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        print(f"✅ RL agent saved to {path_prefix}_actor.pth")
    
    def load(self, path_prefix):
        """Load models"""
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))
        self.is_trained = True
        print(f"✅ RL agent loaded from {path_prefix}_actor.pth")

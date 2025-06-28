import torch


def cross_entropy_dist(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Compute the cross-entropy distance between two probability distributions.

    p and q are tensors of shape (batch_size, num_classes)
    """

    # clamp probabilities to avoid log(0)
    p_clamped = p.clamp(min=eps)
    q_clamped = q.clamp(min=eps)

    term = p_clamped * torch.log(q_clamped) + q_clamped * torch.log(p_clamped)
    return -torch.sum(term, dim=1).mean()

def pairwise_cross_entropy_dist(tensor_list: torch.Tensor, eps: float = 1.0e-8):
    """
    Compute pairwise cross-entropy distances between all tensors in the list.

    ``tensor_list`` is a tensor of shape (num_tensors, batch_size, num_classes)
    """

    # clamp probabilities to avoid log(0)
    tensor_list_clamped = tensor_list.clamp(min=eps)

    # Expand dimensions for broadcasting
    # p: (n, 1, batch_size, num_classes)
    # q: (1, n, batch_size, num_classes)
    p = tensor_list_clamped.unsqueeze(1)
    q = tensor_list_clamped.unsqueeze(0)

    term = p * torch.log(q) + q * torch.log(p)

    # Sum over num_classes dimension, then mean over batch_size dimension
    distances = -torch.sum(term, dim=-1).mean(dim=-1)

    return distances


if __name__ == '__main__':
    # Example usage
    p = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    q = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    print(f"{p = }")
    print(f"{q = }")

    print("Cross-entropy distance:", cross_entropy_dist(p, q))

    tensor_list = torch.stack([p, q])
    print("Pairwise cross-entropy distances:\n", pairwise_cross_entropy_dist(tensor_list))

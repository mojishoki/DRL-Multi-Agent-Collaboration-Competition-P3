def soft_update(target, source, tau):
    """ used for soft update during training """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
def hard_update(target, source):
    """used for copying the local to target parameters during initialization"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

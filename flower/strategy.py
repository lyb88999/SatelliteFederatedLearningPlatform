class SatelliteFedAvg(fl.server.strategy.Strategy):
    def __init__(self,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2):
        """初始化卫星联邦平均策略
        
        Args:
            min_fit_clients: 最小训练客户端数
            min_evaluate_clients: 最小评估客户端数
            min_available_clients: 最小可用客户端数
        """
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients 
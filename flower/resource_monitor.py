import psutil

class ResourceMonitor:
    def get_battery_level(self) -> float:
        """获取电池电量"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
        except:
            pass
        return 100.0  # 如果无法获取电池信息,返回100%
        
    def get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
        
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0 
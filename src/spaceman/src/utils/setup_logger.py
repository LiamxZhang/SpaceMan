import logging

def setup_logger(name, level=logging.INFO):
    """
    设置一个通用的 logger
    
    Args:
        name (str): logger 名称
        level (int): 日志级别，默认为 INFO
        
    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        logger.addHandler(console_handler)
    
    return logger

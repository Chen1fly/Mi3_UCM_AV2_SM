# 设置 / 获取当前查询的自然语言描述
_CURRENT_DESC = None


def set_description(desc: str | None):
    global _CURRENT_DESC;
    _CURRENT_DESC = desc


def get_description(): return _CURRENT_DESC

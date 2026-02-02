"""角色基类和注册器"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional


class Role(ABC):
    """角色基类"""

    name: str = "base"
    description: str = "基础角色"

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词"""
        pass

    def format_user_input(self, text: str) -> str:
        """格式化用户输入（子类可覆盖）"""
        return f"用户输入：{text}"


class RoleRegistry:
    """角色注册表"""

    _roles: Dict[str, Type[Role]] = {}
    _instances: Dict[str, Role] = {}

    @classmethod
    def register(cls, role_class: Type[Role]) -> Type[Role]:
        """注册角色（装饰器）"""
        cls._roles[role_class.name] = role_class
        return role_class

    @classmethod
    def get(cls, name: str) -> Optional[Role]:
        """获取角色实例"""
        if name not in cls._instances:
            role_class = cls._roles.get(name)
            if role_class is None:
                return None
            cls._instances[name] = role_class()
        return cls._instances[name]

    @classmethod
    def list_roles(cls) -> Dict[str, str]:
        """列出所有角色"""
        return {name: role_cls.description for name, role_cls in cls._roles.items()}


def get_role(name: str = "default") -> Role:
    """获取角色，默认返回 default 角色"""
    role = RoleRegistry.get(name)
    if role is None:
        role = RoleRegistry.get("default")
    return role

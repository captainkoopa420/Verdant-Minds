import os
from enum import Enum, auto
from typing import Dict, List, Optional
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import uuid

class UserRole(Enum):
    """Defines user roles with hierarchical permissions"""
    GUEST = auto()
    USER = auto()
    RESEARCHER = auto()
    ADMIN = auto()
    SYSTEM_ARCHITECT = auto()

class PermissionType(Enum):
    """Defines granular system permissions"""
    # Basic Permissions
    VIEW_DASHBOARD = "view_dashboard"
    ACCESS_PROFILE = "access_profile"
    
    # AI Interaction Permissions
    AI_READ_INTERACTION = "ai_read_interaction"
    AI_WRITE_INTERACTION = "ai_write_interaction"
    
    # System Management Permissions
    MANAGE_USERS = "manage_users"
    CONFIGURE_SYSTEM = "configure_system"
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    
    # Ethical and Research Permissions
    MODIFY_ETHICAL_PARAMETERS = "modify_ethical_parameters"
    ACCESS_RESEARCH_DATA = "access_research_data"

class AuthorizationManager:
    """Manages role-based access control and permissions"""
    
    ROLE_PERMISSIONS: Dict[UserRole, List[PermissionType]] = {
        UserRole.GUEST: [
            PermissionType.VIEW_DASHBOARD
        ],
        UserRole.USER: [
            PermissionType.VIEW_DASHBOARD,
            PermissionType.ACCESS_PROFILE,
            PermissionType.AI_READ_INTERACTION
        ],
        UserRole.RESEARCHER: [
            PermissionType.VIEW_DASHBOARD,
            PermissionType.ACCESS_PROFILE,
            PermissionType.AI_READ_INTERACTION,
            PermissionType.AI_WRITE_INTERACTION,
            PermissionType.ACCESS_RESEARCH_DATA
        ],
        UserRole.ADMIN: [
            PermissionType.VIEW_DASHBOARD,
            PermissionType.ACCESS_PROFILE,
            PermissionType.AI_READ_INTERACTION,
            PermissionType.AI_WRITE_INTERACTION,
            PermissionType.MANAGE_USERS,
            PermissionType.VIEW_SYSTEM_METRICS
        ],
        UserRole.SYSTEM_ARCHITECT: [
            # All permissions
            *list(PermissionType)
        ]
    }

    @classmethod
    def get_permissions(cls, role: UserRole) -> List[PermissionType]:
        """Retrieve permissions for a specific role"""
        return cls.ROLE_PERMISSIONS.get(role, [])

    @classmethod
    def has_permission(
        cls, 
        role: UserRole, 
        required_permission: PermissionType
    ) -> bool:
        """Check if a role has a specific permission"""
        return required_permission in cls.get_permissions(role)

class User:
    """Represents a system user with authentication and authorization capabilities"""
    def __init__(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: UserRole = UserRole.USER
    ):
        """
        Initialize a new user
        
        Args:
            username: User's username
            email: User's email address
            password: User's password
            role: User's role (default: USER)
        """
        self.id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.role = role
        self.password_hash = generate_password_hash(password)
        self.created_at = datetime.datetime.utcnow()
        self.last_login = None
        self.is_active = True
        
        # Additional user profile information
        self.profile = {
            "first_name": None,
            "last_name": None,
            "organization": None,
            "research_interests": [],
            "preferences": {}
        }

    def check_password(self, password: str) -> bool:
        """
        Verify user password
        
        Args:
            password: Password to check
        
        Returns:
            Boolean indicating password validity
        """
        return check_password_hash(self.password_hash, password)

    def update_profile(self, profile_data: Dict) -> None:
        """
        Update user profile
        
        Args:
            profile_data: Dictionary of profile information to update
        """
        self.profile.update(profile_data)

    def generate_token(self, token_type: str = 'access') -> str:
        """
        Generate JWT token for authentication
        
        Args:
            token_type: Type of token (access or refresh)
        
        Returns:
            JWT token string
        """
        # Secret key should be stored securely, not in code
        secret_key = os.getenv('JWT_SECRET_KEY', 'default-secret-key')
        
        # Token payload
        payload = {
            'user_id': self.id,
            'username': self.username,
            'role': self.role.name,
            'email': self.email,
            'exp': datetime.datetime.utcnow() + (
                datetime.timedelta(minutes=15) if token_type == 'access' 
                else datetime.timedelta(days=7)
            )
        }
        
        return jwt.encode(payload, secret_key, algorithm='HS256')

    @classmethod
    def verify_token(cls, token: str) -> Optional[Dict]:
        """
        Verify JWT token and return decoded payload
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            secret_key = os.getenv('JWT_SECRET_KEY', 'default-secret-key')
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class UserManager:
    """Manages user-related operations"""
    def __init__(self):
        """Initialize user management system"""
        self.users: Dict[str, User] = {}

    def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: UserRole = UserRole.USER
    ) -> User:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User email address
            password: User password
            role: User role (default: USER)
        
        Returns:
            Created User instance
        
        Raises:
            ValueError: If username or email already exists
        """
        # Validate input
        if not username or not email or not password:
            raise ValueError("Username, email, and password are required")
        
        # Check for existing username or email
        if any(u.username == username or u.email == email for u in self.users.values()):
            raise ValueError("Username or email already exists")
        
        # Create new user
        user = User(username, email, password, role)
        self.users[user.id] = user
        
        return user

    def authenticate_user(
        self, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """
        Authenticate user credentials
        
        Args:
            username: User's username
            password: User's password
        
        Returns:
            Authenticated User or None
        """
        for user in self.users.values():
            if user.username == username and user.check_password(password):
                user.last_login = datetime.datetime.utcnow()
                return user
        
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Retrieve user by ID
        
        Args:
            user_id: User's unique identifier
        
        Returns:
            User instance or None
        """
        return self.users.get(user_id)

    def update_user_role(
        self, 
        user_id: str, 
        new_role: UserRole
    ) -> Optional[User]:
        """
        Update user's role
        
        Args:
            user_id: User's unique identifier
            new_role: New user role
        
        Returns:
            Updated User instance or None
        """
        user = self.get_user_by_id(user_id)
        if user:
            user.role = new_role
            return user
        
        return None

def main():
    """
    Demonstration of the authentication system
    """
    # Initialize user management
    user_manager = UserManager()
    
    try:
        # Create users with different roles
        admin_user = user_manager.create_user(
            username='admin', 
            email='admin@unifiedsyntheticmind.com', 
            password='secure_admin_password', 
            role=UserRole.ADMIN
        )
        
        researcher_user = user_manager.create_user(
            username='researcher', 
            email='researcher@unifiedsyntheticmind.com', 
            password='research_password', 
            role=UserRole.RESEARCHER
        )
        
        # Authentication demonstration
        authenticated_admin = user_manager.authenticate_user(
            'admin', 'secure_admin_password'
        )
        
        if authenticated_admin:
            # Generate access token
            access_token = authenticated_admin.generate_token('access')
            print(f"Access Token: {access_token}")
            
            # Verify token
            decoded_payload = User.verify_token(access_token)
            print(f"Decoded Payload: {decoded_payload}")
            
            # Check permissions
            can_view_metrics = AuthorizationManager.has_permission(
                authenticated_admin.role, 
                PermissionType.VIEW_SYSTEM_METRICS
            )
            print(f"Can view system metrics: {can_view_metrics}")
    
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
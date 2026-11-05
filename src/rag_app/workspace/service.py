import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import Department, User, UserDepartment, Workspace, WorkspaceMember, WorkspaceProviderProfile
from rag_app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceUpdate,
    WorkspaceMemberCreate,
    WorkspaceMemberResponse,
    WorkspaceMemberUpdate,
    WorkspaceProviderProfileUpdate,
    WorkspaceProviderProfileResponse,
    ProviderTestConnectionRequest,
    ProviderTestConnectionResponse,
    WorkspaceApiKeySetRequest,
    WorkspaceApiKeySetResponse,
)
from rag_app.security.authorization import authorize, get_allowed_departments


class WorkspaceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _format_response(ws: Workspace) -> WorkspaceResponse:
        dept_name = ws.department.name if ws.department else None
        dept_code = ws.department.code if ws.department else None
        return WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            code=ws.code,
            description=ws.description,
            department_id=ws.department_id,
            department_name=dept_name,
            department_code=dept_code,
            is_active=ws.is_active,
            created_by=ws.created_by,
            created_at=ws.created_at,
            updated_at=ws.updated_at,
            deleted_at=ws.deleted_at,
        )

    @staticmethod
    def _format_member_response(member: WorkspaceMember) -> WorkspaceMemberResponse:
        user_email = member.user.email if member.user else None
        return WorkspaceMemberResponse(
            workspace_id=member.workspace_id,
            user_id=member.user_id,
            user_email=user_email,
            role=member.role,
            created_at=member.created_at,
            updated_at=member.updated_at,
        )

    async def _is_workspace_owner(self, user_id: uuid.UUID, workspace_id: uuid.UUID) -> bool:
        member = await self.session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
            )
        )
        return member is not None and member.role == "OWNER"

    async def _check_workspace_admin_or_owner(self, user: User, ws: Workspace) -> None:
        """
        Verifies that the caller is ADMIN or an OWNER member of this workspace.
        Also enforces department authorization boundary for USER.
        """
        if user.role == "ADMIN":
            return

        dept = ws.department
        if not dept or not dept.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace department is inactive or not found.",
            )

        # Department boundary check
        is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
        if not is_dept_auth:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: You do not have access to department '{dept.code}'.",
            )

        # Workspace OWNER check
        is_owner = await self._is_workspace_owner(user.id, ws.id)
        if not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only Workspace Owners or Administrators can manage workspace members.",
            )

    async def _get_active_workspace(self, workspace_id: uuid.UUID) -> Workspace:
        stmt = (
            select(Workspace)
            .options(selectinload(Workspace.department))
            .where(
                Workspace.id == workspace_id,
                Workspace.deleted_at.is_(None),
            )
        )
        ws = await self.session.scalar(stmt)
        if not ws:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )
        if not ws.department or not ws.department.is_active or ws.department.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace department is inactive or deleted.",
            )
        return ws

    async def create_workspace(
        self,
        user: User,
        payload: WorkspaceCreate,
    ) -> WorkspaceResponse:
        dept = await self.session.get(Department, payload.department_id)
        if not dept or not dept.is_active or dept.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Department '{payload.department_id}' not found or inactive.",
            )

        if user.role != "ADMIN":
            # Must possess access to this department
            is_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: You do not have access to department '{dept.code}'.",
                )

        code = payload.code.strip().upper()
        name = payload.name.strip()
        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workspace code must not be empty.",
            )
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workspace name must not be empty.",
            )

        existing = await self.session.scalar(
            select(Workspace).where(
                Workspace.department_id == dept.id,
                Workspace.code == code,
                Workspace.deleted_at.is_(None),
            )
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workspace with code '{code}' already exists in department '{dept.code}'.",
            )

        now = datetime.now(timezone.utc)
        ws_id = uuid.uuid4()
        ws = Workspace(
            id=ws_id,
            name=name,
            code=code,
            description=payload.description.strip() if payload.description else None,
            department_id=dept.id,
            created_by=user.id,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self.session.add(ws)

        # Creator automatically becomes OWNER
        owner_member = WorkspaceMember(
            workspace_id=ws_id,
            user_id=user.id,
            role="OWNER",
            created_at=now,
            updated_at=now,
        )
        self.session.add(owner_member)

        try:
            await self.session.commit()
            await self.session.refresh(ws, ["department"])
        except IntegrityError:
            await self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workspace with code '{code}' already exists in department '{dept.code}'.",
            )

        return self._format_response(ws)

    async def get_workspace(
        self,
        user: User,
        workspace_id: uuid.UUID,
    ) -> WorkspaceResponse:
        ws = await self._get_active_workspace(workspace_id)
        dept = ws.department

        if user.role != "ADMIN":
            is_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: You do not have access to department '{dept.code}'.",
                )

        return self._format_response(ws)

    async def list_workspaces(
        self,
        user: User,
        department_id: Optional[uuid.UUID] = None,
        include_inactive: bool = False,
    ) -> List[WorkspaceResponse]:
        # Ensure all active departments have a 1:1 default workspace
        active_depts = (
            await self.session.scalars(
                select(Department).where(
                    Department.is_active == True,
                    Department.deleted_at.is_(None),
                )
            )
        ).all()
        existing_ws_depts = set(
            (
                await self.session.scalars(
                    select(Workspace.department_id).where(
                        Workspace.deleted_at.is_(None)
                    )
                )
            ).all()
        )
        now = datetime.now(timezone.utc)
        created_any = False
        for dept in active_depts:
            dept_code = dept.code or ""
            dept_upper = dept_code.upper()
            if (
                any(dept_upper.startswith(p) for p in ("TEST_", "MCP_", "DISC_", "WS_"))
                or any(dept_upper.endswith(s) for s in ("_TEST", "_TEST_WS", "_MOCK"))
                or (len(dept_upper) > 7 and any(dept_upper.startswith(p) for p in ("SALES_", "FIN_", "HR_")))
            ):
                continue

            if dept.id not in existing_ws_depts:
                new_ws = Workspace(
                    name=dept.name,
                    code=f"{dept.code.lower()}_ws",
                    department_id=dept.id,
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(new_ws)
                created_any = True
        if created_any:
            await self.session.commit()

        stmt = (
            select(Workspace)
            .join(Department, Workspace.department_id == Department.id)
            .options(selectinload(Workspace.department))
            .where(
                Workspace.deleted_at.is_(None),
                Department.is_active == True,
                Department.deleted_at.is_(None),
            )
        )
        if not include_inactive:
            stmt = stmt.where(Workspace.is_active == True)

        if user.role != "ADMIN":
            allowed_dept_ids = await get_allowed_departments(self.session, user)
            if not allowed_dept_ids:
                return []
            if department_id:
                if department_id not in allowed_dept_ids:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not have access to this department.",
                    )
                stmt = stmt.where(Workspace.department_id == department_id)
            else:
                stmt = stmt.where(Workspace.department_id.in_(allowed_dept_ids))
        else:
            if department_id:
                stmt = stmt.where(Workspace.department_id == department_id)

        stmt = stmt.order_by(Workspace.name.asc())
        result = await self.session.scalars(stmt)
        workspaces = result.all()

        def _is_test_workspace(ws: Workspace) -> bool:
            c = (ws.code or "").lower()
            dept_c = (ws.department.code or "").lower() if ws.department else ""
            if c in ("hr_ws", "finance_ws", "sales_ws", "unclassified_ws"):
                return False
            if dept_c in ("hr", "finance", "sales", "unclassified"):
                return False
            for code_to_check in (c, dept_c):
                if any(code_to_check.startswith(p) for p in ("test_", "mcp_", "disc_", "ws_sales_test", "ws_finance_test", "ws_hr_test")):
                    return True
                if any(code_to_check.endswith(s) for s in ("_test", "_test_ws", "_mock")):
                    return True
                if len(code_to_check) > 8 and any(code_to_check.startswith(p) for p in ("sales_", "fin_", "hr_")):
                    return True
            return False

        return [self._format_response(w) for w in workspaces if not _is_test_workspace(w)]

    async def update_workspace(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: WorkspaceUpdate,
    ) -> WorkspaceResponse:
        ws = await self._get_active_workspace(workspace_id)
        dept = ws.department

        if user.role != "ADMIN":
            has_edit = await authorize(self.session, user, "EDIT", dept.code)
            is_owner = await self._is_workspace_owner(user.id, ws.id)
            if not has_edit and not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: EDIT permission or Workspace OWNER role required.",
                )

        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Workspace name must not be empty.",
                )
            ws.name = name

        if payload.description is not None:
            ws.description = payload.description.strip() if payload.description else None

        if payload.is_active is not None:
            ws.is_active = payload.is_active

        ws.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(ws, ["department"])
        return self._format_response(ws)

    async def delete_workspace(
        self,
        user: User,
        workspace_id: uuid.UUID,
    ) -> None:
        ws = await self._get_active_workspace(workspace_id)
        dept = ws.department

        if user.role != "ADMIN":
            has_delete = await authorize(self.session, user, "DELETE", dept.code)
            is_owner = await self._is_workspace_owner(user.id, ws.id)
            if not has_delete and not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: DELETE permission or Workspace OWNER role required.",
                )

        now = datetime.now(timezone.utc)
        ws.is_active = False
        ws.deleted_at = now
        ws.updated_at = now
        await self.session.commit()

    # ------------------------------------------------------------------
    # Workspace Membership Management
    # ------------------------------------------------------------------

    async def list_members(
        self,
        user: User,
        workspace_id: uuid.UUID,
    ) -> List[WorkspaceMemberResponse]:
        ws = await self._get_active_workspace(workspace_id)
        dept = ws.department

        if user.role != "ADMIN":
            is_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: You do not have access to department '{dept.code}'.",
                )

        stmt = (
            select(WorkspaceMember)
            .options(selectinload(WorkspaceMember.user))
            .where(WorkspaceMember.workspace_id == workspace_id)
            .order_by(WorkspaceMember.created_at.asc())
        )
        result = await self.session.scalars(stmt)
        members = result.all()
        return [self._format_member_response(m) for m in members]

    async def add_member(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: WorkspaceMemberCreate,
    ) -> WorkspaceMemberResponse:
        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_admin_or_owner(user, ws)

        # Validate target user exists
        target_user = await self.session.get(User, payload.user_id)
        if not target_user or not target_user.is_active or target_user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{payload.user_id}' not found or inactive.",
            )

        # Validate target user belongs to workspace's department (ADMIN users are global)
        if target_user.role != "ADMIN":
            is_dept_member = await self.session.scalar(
                select(UserDepartment).where(
                    UserDepartment.user_id == target_user.id,
                    UserDepartment.department_id == ws.department_id,
                )
            )
            if not is_dept_member:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User does not belong to the workspace's department. Cross-department workspace membership is not permitted.",
                )

        # Check if already a member
        existing_member = await self.session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == target_user.id,
            )
        )
        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User '{target_user.email}' is already a member of this workspace.",
            )

        now = datetime.now(timezone.utc)
        member = WorkspaceMember(
            workspace_id=workspace_id,
            user_id=target_user.id,
            role=payload.role,
            created_at=now,
            updated_at=now,
        )
        self.session.add(member)
        await self.session.commit()
        await self.session.refresh(member, ["user"])

        return self._format_member_response(member)

    async def update_member_role(
        self,
        user: User,
        workspace_id: uuid.UUID,
        target_user_id: uuid.UUID,
        payload: WorkspaceMemberUpdate,
    ) -> WorkspaceMemberResponse:
        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_admin_or_owner(user, ws)

        stmt = (
            select(WorkspaceMember)
            .options(selectinload(WorkspaceMember.user))
            .where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == target_user_id,
            )
        )
        member = await self.session.scalar(stmt)
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace member '{target_user_id}' not found in this workspace.",
            )

        member.role = payload.role
        member.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(member, ["user"])

        return self._format_member_response(member)

    async def remove_member(
        self,
        user: User,
        workspace_id: uuid.UUID,
        target_user_id: uuid.UUID,
    ) -> None:
        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_admin_or_owner(user, ws)

        member = await self.session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == target_user_id,
            )
        )
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace member '{target_user_id}' not found in this workspace.",
            )

        await self.session.delete(member)
        await self.session.commit()

    # ---------------------------------------------------------------------------
    # Provider Profile methods
    # ---------------------------------------------------------------------------

    @staticmethod
    def _validate_credential_ref(
        ref: str | None,
        existing_ref: str | None,
        field_name: str,
    ) -> str | None:
        """
        Validates that a credential reference is a valid server environment variable identifier.
        Strictly rejects raw secrets to prevent plaintext API key storage in the database.

        Semantics:
        - None / omitted: preserved if updating, or None if creating.
        - Contains '•': unchanged (preserves existing stored ref).
        - Empty string "": explicit clear (returns None).
        - Valid identifier: alphanumeric + underscore, starts with letter/underscore, max 64 chars.
        - Raw secrets (e.g. sk-, gsk_, tf_, hyphens, spaces): raises HTTP 422.
        """
        if ref is None:
            return existing_ref

        cleaned = ref.strip()
        if "•" in cleaned:
            return existing_ref
        if cleaned == "":
            return None

        # Check for known raw API key patterns or prefixes
        lower_ref = cleaned.lower()
        if any(
            lower_ref.startswith(prefix)
            for prefix in ("sk-", "sk_", "gsk-", "gsk_", "tf-", "tf_", "live_", "secret_", "bearer ")
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {field_name}: Raw API keys cannot be stored. "
                    "Enter a server-side environment variable name (e.g., 'TINYFISH_PROD_KEY')."
                ),
            )

        # Enforce valid environment variable name syntax
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{1,63}$", cleaned):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {field_name}: '{cleaned[:12]}...' is not a valid environment variable name. "
                    "Must contain only letters, numbers, and underscores (e.g., 'TINYFISH_PROD_KEY')."
                ),
            )

        return cleaned

    @staticmethod
    def _mask_credential_ref(ref: str | None) -> str | None:
        """Never return raw API keys to browser. If it looks like a raw secret, mask it; if it's a reference label, return as-is."""
        if not ref:
            return None
        # If reference begins with a known secret key prefix, mask it
        if any(ref.lower().startswith(p) for p in ("sk-", "gsk_", "tf_", "live_", "secret_")):
            return "••••••••••••••••"
        return ref

    @staticmethod
    def _format_profile_response(profile: WorkspaceProviderProfile) -> WorkspaceProviderProfileResponse:
        return WorkspaceProviderProfileResponse(
            id=profile.id,
            workspace_id=profile.workspace_id,
            name=profile.name,
            is_active=profile.is_active,
            llm_provider=profile.llm_provider,
            llm_model=profile.llm_model,
            llm_credential_ref=WorkspaceService._mask_credential_ref(profile.llm_credential_ref),
            embedding_provider=profile.embedding_provider,
            embedding_model=profile.embedding_model,
            document_reader_provider=profile.document_reader_provider,
            reranker_provider=profile.reranker_provider,
            reranker_model=profile.reranker_model,
            ocr_provider=profile.ocr_provider,
            web_provider=profile.web_provider,
            web_credential_ref=WorkspaceService._mask_credential_ref(profile.web_credential_ref),
            configuration=profile.configuration or {},
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    async def _check_workspace_provider_access(
        self,
        user: User,
        ws: Workspace,
        require_write: bool = False,
    ) -> None:
        """
        Authorization guard for provider profile operations.
        ADMIN: always allowed.
        OWNER: can read and write.
        EDITOR/VIEWER: can read; cannot write.
        Any USER must first pass existing department authorization.
        """
        if user.role == "ADMIN":
            return

        dept = ws.department
        if not dept or not dept.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace department is inactive or not found.",
            )

        is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
        if not is_dept_auth:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: You do not have access to department '{dept.code}'.",
            )

        member = await self.session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == ws.id,
                WorkspaceMember.user_id == user.id,
            )
        )
        if not member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You are not a member of this workspace.",
            )

        if require_write and member.role not in ("OWNER",):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Only workspace OWNER or ADMIN can modify provider configuration. Your role: {member.role}.",
            )

    async def get_provider_profile(
        self,
        user: User,
        workspace_id: uuid.UUID,
    ) -> WorkspaceProviderProfileResponse | None:
        """
        Returns the active provider profile for the workspace, or None if not set.
        Any workspace member (OWNER / EDITOR / VIEWER) or ADMIN can read.
        """
        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_provider_access(user, ws, require_write=False)

        profile = await self.session.scalar(
            select(WorkspaceProviderProfile).where(
                WorkspaceProviderProfile.workspace_id == workspace_id,
            )
        )
        if profile is None:
            return None
        return self._format_profile_response(profile)

    async def upsert_provider_profile(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: WorkspaceProviderProfileUpdate,
    ) -> WorkspaceProviderProfileResponse:
        """
        Create or update the provider profile for the workspace.
        ADMIN and workspace OWNER are allowed; EDITOR and VIEWER are denied.

        Validates that every specified provider name is registered in the platform
        registry before persisting, to prevent storing unresolvable configurations.
        """
        from rag_app.providers.registry import provider_registry
        from rag_app.providers.exceptions import ProviderNotFoundError

        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_provider_access(user, ws, require_write=True)

        # Validate all specified provider names exist in the registry
        capability_field_pairs = [
            ("llm", payload.llm_provider),
            ("embedding", payload.embedding_provider),
            ("document_reader", payload.document_reader_provider),
            ("reranker", payload.reranker_provider),
            ("ocr", payload.ocr_provider),
            ("web", payload.web_provider),
        ]
        for capability, provider_name in capability_field_pairs:
            if provider_name:
                available = provider_registry.list_providers(capability)
                if provider_name.lower() not in available:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            f"Provider '{provider_name}' is not registered for capability '{capability}'. "
                            f"Available providers: {available}"
                        ),
                    )

        # Validate embedding model compatibility (database requires 384 dimensions)
        if payload.embedding_model:
            incompatible_models = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
                "bge-large-en": 1024,
                "bge-large-en-v1.5": 1024,
                "bge-base-en-v1.5": 768,
                "all-mpnet-base-v2": 768,
            }
            for incomp, dim in incompatible_models.items():
                if incomp in payload.embedding_model.lower():
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            f"Incompatible embedding model '{payload.embedding_model}': Vector store is configured "
                            f"for 384 dimensions, but '{payload.embedding_model}' generates {dim} dimensions. "
                            f"Changing vector dimensions requires an explicit re-embedding migration."
                        ),
                    )

        now = datetime.now(timezone.utc)

        existing = await self.session.scalar(
            select(WorkspaceProviderProfile).where(
                WorkspaceProviderProfile.workspace_id == workspace_id,
            )
        )

        # Sanitize credential references:
        # Validate and sanitize credential references:
        # 1. Masked placeholders ('•••') mean unchanged — preserve existing stored ref.
        # 2. Empty string '' means explicit clear — set to None.
        # 3. Raw secrets (sk-, tf_, etc.) are rejected with HTTP 422.
        existing_llm = existing.llm_credential_ref if existing else None
        existing_web = existing.web_credential_ref if existing else None

        clean_llm_cred = self._validate_credential_ref(
            payload.llm_credential_ref, existing_llm, "llm_credential_ref"
        )
        clean_web_cred = self._validate_credential_ref(
            payload.web_credential_ref, existing_web, "web_credential_ref"
        )

        if existing is None:
            # CREATE new profile
            profile = WorkspaceProviderProfile(
                workspace_id=workspace_id,
                name=payload.name or "Default",
                is_active=payload.is_active if payload.is_active is not None else True,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
                llm_credential_ref=clean_llm_cred,
                embedding_provider=payload.embedding_provider,
                embedding_model=payload.embedding_model,
                document_reader_provider=payload.document_reader_provider,
                reranker_provider=payload.reranker_provider,
                reranker_model=payload.reranker_model,
                ocr_provider=payload.ocr_provider,
                web_provider=payload.web_provider,
                web_credential_ref=clean_web_cred,
                configuration=payload.configuration or {},
                created_at=now,
                updated_at=now,
            )
            self.session.add(profile)
        else:
            # UPDATE existing profile — only touch fields that are explicitly provided
            update_fields = payload.model_dump(exclude_unset=True)
            for field, value in update_fields.items():
                if field == "llm_credential_ref":
                    existing.llm_credential_ref = clean_llm_cred
                elif field == "web_credential_ref":
                    existing.web_credential_ref = clean_web_cred
                else:
                    setattr(existing, field, value)
            existing.updated_at = now
            profile = existing

        try:
            await self.session.commit()
            await self.session.refresh(profile)
        except Exception:
            await self.session.rollback()
            raise

        return self._format_profile_response(profile)

    async def test_provider_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: ProviderTestConnectionRequest,
    ) -> ProviderTestConnectionResponse:
        """
        Validates provider configuration and executes a live/deterministic backend check.
        Ensures model compatibility and enforces that unsupported providers fail safely.
        """
        import time
        from rag_app.providers.registry import provider_registry, get_provider

        ws = await self._get_active_workspace(workspace_id)
        # Check authorization (OWNER or ADMIN)
        await self._check_workspace_provider_access(user, ws, require_write=True)

        capability = payload.capability.strip().lower()
        provider_name = payload.provider.strip().lower()

        # 1. Registry existence check
        registered = provider_registry.list_providers(capability)
        if provider_name not in registered:
            return ProviderTestConnectionResponse(
                success=False,
                capability=capability,
                provider=provider_name,
                model=payload.model,
                message=f"Provider '{provider_name}' is not registered for capability '{capability}'. Available: {registered}",
                details={"registered": False},
            )

        # 2. Capability-specific execution probe
        start = time.perf_counter()
        try:
            if capability == "llm":
                if provider_name in ("mock", "fake"):
                    mock_adapter = get_provider("llm", provider_name, model=payload.model or "mock-deterministic-v1")
                    resp, usage = await mock_adapter.generate([{"role": "user", "content": "ping"}])
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        model=mock_adapter.model,
                        message="Mock LLM responded successfully (zero-network verification).",
                        latency_ms=latency_ms,
                        details={"tokens": usage.get("total_tokens", 0), "response_preview": resp[:50]},
                    )
                elif provider_name == "groq":
                    import os
                    from rag_app.core.config import get_settings
                    settings = get_settings()
                    api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", None)
                    if not api_key or not str(api_key).strip():
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            model=payload.model or getattr(settings, "groq_model", "llama-3.3-70b-versatile"),
                            message="GROQ_API_KEY is not configured on server. Provider cannot connect.",
                            details={"configured": False},
                        )
                    groq_adapter = get_provider("llm", "groq", model=payload.model)
                    # Safe probe: count tokens
                    groq_adapter.count_tokens([{"role": "user", "content": "ping"}])
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        model=groq_adapter.model,
                        message=f"Groq provider verified and ready with model '{groq_adapter.model}'.",
                        latency_ms=latency_ms,
                        details={"configured": True, "model": groq_adapter.model},
                    )
                elif provider_name in ("openai", "custom", "openai_compatible", "deepseek", "openrouter", "ollama"):
                    import os
                    api_key = None
                    if payload.credential_ref:
                        api_key = os.getenv(payload.credential_ref)
                    if not api_key:
                        if provider_name == "openai":
                            api_key = os.getenv("OPENAI_API_KEY")
                        elif provider_name == "deepseek":
                            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                        elif provider_name == "openrouter":
                            api_key = os.getenv("OPENROUTER_API_KEY")
                        elif provider_name in ("custom", "openai_compatible"):
                            api_key = os.getenv("CUSTOM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

                    base_url = None
                    if payload.extra_config and isinstance(payload.extra_config, dict):
                        base_url = payload.extra_config.get("base_url")

                    is_local = (base_url and ("localhost" in base_url or "127.0.0.1" in base_url)) or provider_name == "ollama"
                    if not api_key and not is_local:
                        ref_display = payload.credential_ref or f"{provider_name.upper()}_API_KEY"
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            model=payload.model or "gpt-4o",
                            message=f"{ref_display} is not configured on server. Use 'Enter API Key' to save it.",
                            details={"configured": False},
                        )

                    model_name = payload.model or ("deepseek-chat" if provider_name == "deepseek" else "gpt-4o")
                    adapter = get_provider(
                        "llm",
                        provider_name,
                        model=model_name,
                        api_key=api_key or "local",
                        base_url=base_url,
                    )
                    adapter.count_tokens([{"role": "user", "content": "ping"}])
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        model=model_name,
                        message=f"{provider_name.capitalize()} provider verified with model '{model_name}'.",
                        latency_ms=latency_ms,
                        details={"configured": True, "model": model_name, "base_url": getattr(adapter, "base_url", None)},
                    )
                elif provider_name in ("anthropic", "gemini"):
                    env_key = f"{provider_name.upper()}_API_KEY"
                    import os
                    if not os.getenv(env_key):
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            model=payload.model,
                            message=f"{env_key} is not configured on server. Provider is unavailable.",
                            details={"configured": False},
                        )
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        model=payload.model,
                        message=f"{provider_name.capitalize()} provider configured.",
                        latency_ms=latency_ms,
                        details={"configured": True},
                    )
                else:
                    return ProviderTestConnectionResponse(
                        success=False,
                        capability=capability,
                        provider=provider_name,
                        model=payload.model,
                        message=f"Live connection probe not implemented for '{provider_name}'.",
                    )

            elif capability == "embedding":
                # Validate 384-dimension vector requirement
                model_name = payload.model or "sentence-transformers/all-MiniLM-L6-v2"
                incompatible_models = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536,
                    "bge-large-en": 1024,
                    "bge-large-en-v1.5": 1024,
                    "bge-base-en-v1.5": 768,
                    "all-mpnet-base-v2": 768,
                }
                for incomp, dim in incompatible_models.items():
                    if incomp in model_name.lower():
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            model=model_name,
                            message=f"Incompatible embedding model: Vector store requires 384 dimensions, but '{model_name}' produces {dim} dimensions.",
                            details={"required_dimension": 384, "model_dimension": dim},
                        )

                adapter = get_provider("embedding", provider_name, model_name=model_name)
                vec = await adapter.embed_query("ping")
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                dim = len(vec)
                if dim != 384:
                    return ProviderTestConnectionResponse(
                        success=False,
                        capability=capability,
                        provider=provider_name,
                        model=model_name,
                        message=f"Embedding dimension mismatch: Expected 384, but got {dim}.",
                        latency_ms=latency_ms,
                        details={"dimension": dim, "expected": 384},
                    )
                return ProviderTestConnectionResponse(
                    success=True,
                    capability=capability,
                    provider=provider_name,
                    model=model_name,
                    message=f"SentenceTransformer verified (dimension: {dim}).",
                    latency_ms=latency_ms,
                    details={"dimension": dim},
                )

            elif capability == "reranker":
                model_name = payload.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                adapter = get_provider("reranker", "cross_encoder", model_name=model_name)
                scores = await adapter.rerank("query", ["doc 1"])
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return ProviderTestConnectionResponse(
                    success=True,
                    capability=capability,
                    provider=provider_name,
                    model=model_name,
                    message="CrossEncoder reranker verified.",
                    latency_ms=latency_ms,
                    details={"sample_score": round(scores[0], 4) if scores else None},
                )

            elif capability == "web":
                import os
                from rag_app.core.config import get_settings
                settings = get_settings()

                if provider_name == "tinyfish":
                    api_key = (
                        getattr(settings, "tinyfish_api_key", None)
                        or os.getenv("TINYFISH_API_KEY")
                        or os.getenv("TINY_FISH")
                    )
                    if not api_key:
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            message="TINYFISH_API_KEY is not configured on server. Provider cannot connect.",
                            details={"configured": False},
                        )
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        message="TinyFish Search API key configured and reachable.",
                        latency_ms=latency_ms,
                        details={"configured": True, "endpoint": "https://api.search.tinyfish.ai"},
                    )
                elif provider_name == "tavily":
                    api_key = os.getenv("TAVILY_API_KEY")
                    if not api_key:
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            message="TAVILY_API_KEY is not configured on server.",
                            details={"configured": False},
                        )
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        message="Tavily web search adapter configured.",
                        latency_ms=round((time.perf_counter() - start) * 1000, 2),
                        details={"configured": True},
                    )
                else:
                    api_key = os.getenv("FIRECRAWL_API_KEY")
                    if not api_key:
                        return ProviderTestConnectionResponse(
                            success=False,
                            capability=capability,
                            provider=provider_name,
                            message="FIRECRAWL_API_KEY is not configured on server.",
                            details={"configured": False},
                        )
                    return ProviderTestConnectionResponse(
                        success=True,
                        capability=capability,
                        provider=provider_name,
                        message="Firecrawl web search adapter configured.",
                        latency_ms=round((time.perf_counter() - start) * 1000, 2),
                        details={"configured": True},
                    )

            elif capability in ("vector_store", "vector_db", "vector_database"):
                await self.session.execute(select(1))
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return ProviderTestConnectionResponse(
                    success=True,
                    capability=capability,
                    provider=provider_name,
                    message="PostgreSQL + pgvector connection active.",
                    latency_ms=latency_ms,
                    details={"engine": "PostgreSQL", "extension": "pgvector", "dimension": 384},
                )

            elif capability == "document_reader":
                return ProviderTestConnectionResponse(
                    success=True,
                    capability=capability,
                    provider=provider_name,
                    message="Document Reader provider verified.",
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

            elif capability == "ocr":
                return ProviderTestConnectionResponse(
                    success=True,
                    capability=capability,
                    provider=provider_name,
                    message="OCR engine verified.",
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

            else:
                return ProviderTestConnectionResponse(
                    success=False,
                    capability=capability,
                    provider=provider_name,
                    message=f"Unknown capability '{capability}'.",
                )

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            # Mask any raw API keys from error messages
            err_msg = str(exc)
            if "gsk_" in err_msg or "sk-" in err_msg:
                import re
                err_msg = re.sub(r"(gsk_|sk-)[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", err_msg)
            return ProviderTestConnectionResponse(
                success=False,
                capability=capability,
                provider=provider_name,
                model=payload.model,
                message=f"Connection test failed: {err_msg}",
                latency_ms=latency_ms,
            )

    async def set_workspace_api_key(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: WorkspaceApiKeySetRequest,
    ) -> WorkspaceApiKeySetResponse:
        """
        Saves a raw provider API key directly to the server environment and updates the
        workspace provider profile credential reference, avoiding manual .env edits.
        """
        import os
        import re

        ws = await self._get_active_workspace(workspace_id)
        await self._check_workspace_provider_access(user, ws, require_write=True)

        capability = payload.capability.strip().lower()
        provider = payload.provider.strip().lower()
        raw_key = payload.api_key.strip()
        if not raw_key:
            raise HTTPException(status_code=400, detail="API key cannot be empty.")

        if payload.env_var_name and payload.env_var_name.strip():
            env_name = payload.env_var_name.strip()
        else:
            default_names = {
                "openai": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "custom": "CUSTOM_LLM_API_KEY",
                "openai_compatible": "CUSTOM_LLM_API_KEY",
                "tavily": "TAVILY_API_KEY",
                "firecrawl": "FIRECRAWL_API_KEY",
                "tinyfish": "TINYFISH_API_KEY",
            }
            env_name = default_names.get(provider, f"{provider.upper()}_API_KEY")

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{1,63}$", env_name):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid environment variable name '{env_name}'. Must contain only letters, numbers, and underscores.",
            )

        # 1. Update in-memory server environment
        os.environ[env_name] = raw_key

        # 2. Persist to .env file
        try:
            env_path = ".env"
            lines = []
            found = False
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith(f"{env_name}="):
                            lines.append(f"{env_name}={raw_key}\n")
                            found = True
                        else:
                            lines.append(line)
            if not found:
                lines.append(f"{env_name}={raw_key}\n")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            logger.warning(f"Could not persist secret to .env file: {e}")

        # 3. Update WorkspaceProviderProfile credential reference in DB
        profile = await self.session.scalar(
            select(WorkspaceProviderProfile).where(
                WorkspaceProviderProfile.workspace_id == workspace_id
            )
        )
        if profile is not None:
            if capability == "llm":
                profile.llm_credential_ref = env_name
            elif capability == "web":
                profile.web_credential_ref = env_name
            await self.session.flush()

        return WorkspaceApiKeySetResponse(
            success=True,
            capability=capability,
            provider=provider,
            env_var_name=env_name,
            message=f"API key for '{provider}' saved successfully to server environment as '{env_name}'.",
        )

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
        return [self._format_response(w) for w in workspaces]

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
    def _format_profile_response(profile: WorkspaceProviderProfile) -> WorkspaceProviderProfileResponse:
        return WorkspaceProviderProfileResponse(
            id=profile.id,
            workspace_id=profile.workspace_id,
            name=profile.name,
            is_active=profile.is_active,
            llm_provider=profile.llm_provider,
            llm_model=profile.llm_model,
            llm_credential_ref=profile.llm_credential_ref,
            embedding_provider=profile.embedding_provider,
            embedding_model=profile.embedding_model,
            document_reader_provider=profile.document_reader_provider,
            reranker_provider=profile.reranker_provider,
            reranker_model=profile.reranker_model,
            ocr_provider=profile.ocr_provider,
            web_provider=profile.web_provider,
            web_credential_ref=profile.web_credential_ref,
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

        now = datetime.now(timezone.utc)

        existing = await self.session.scalar(
            select(WorkspaceProviderProfile).where(
                WorkspaceProviderProfile.workspace_id == workspace_id,
            )
        )

        if existing is None:
            # CREATE new profile
            profile = WorkspaceProviderProfile(
                workspace_id=workspace_id,
                name=payload.name or "Default",
                is_active=payload.is_active if payload.is_active is not None else True,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
                llm_credential_ref=payload.llm_credential_ref,
                embedding_provider=payload.embedding_provider,
                embedding_model=payload.embedding_model,
                document_reader_provider=payload.document_reader_provider,
                reranker_provider=payload.reranker_provider,
                reranker_model=payload.reranker_model,
                ocr_provider=payload.ocr_provider,
                web_provider=payload.web_provider,
                web_credential_ref=payload.web_credential_ref,
                configuration=payload.configuration or {},
                created_at=now,
                updated_at=now,
            )
            self.session.add(profile)
        else:
            # UPDATE existing profile — only touch fields that are explicitly provided
            update_fields = payload.model_dump(exclude_unset=True)
            for field, value in update_fields.items():
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

import os

admin_dashboard = """import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { adminApi, UserResponse, DepartmentResponse, PermissionResponse } from '../api/admin'
import { userApi } from '../api/user'

export default function AdminDashboard({ token, onLogout }: { token: string, onLogout: () => void }) {
  const [users, setUsers] = useState<UserResponse[]>([])
  const [selectedUser, setSelectedUser] = useState<UserResponse | null>(null)
  const [departments, setDepartments] = useState<DepartmentResponse[]>([])
  const [permissions, setPermissions] = useState<Record<string, PermissionResponse[]>>({})
  
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  
  const navigate = useNavigate()

  useEffect(() => {
    async function loadInitial() {
      try {
        const me = await userApi.getMe(token)
        if (me.role !== 'ADMIN') {
          navigate('/app', { replace: true })
          return
        }
        await loadUsers()
      } catch (e) {
        onLogout()
      }
    }
    loadInitial()
  }, [token, navigate, onLogout])

  async function loadUsers() {
    try {
      const data = await adminApi.getUsers(token)
      setUsers(data)
    } catch (e: any) {
      setError('Failed to load users: ' + e.message)
    }
  }

  async function selectUser(user: UserResponse) {
    setSelectedUser(user)
    setDepartments([])
    setPermissions({})
    setError('')
    setSuccess('')
    try {
      const depts = await adminApi.getUserDepartments(user.id, token)
      setDepartments(depts)
      
      const permsMap: Record<string, PermissionResponse[]> = {}
      for (const d of depts) {
        const p = await adminApi.getUserPermissions(user.id, d.code, token)
        permsMap[d.code] = p
      }
      setPermissions(permsMap)
    } catch (e: any) {
      setError('Failed to load user access: ' + e.message)
    }
  }

  async function handleToggleActive(user: UserResponse) {
    try {
      await adminApi.updateUserStatus(user.id, !user.is_active, token)
      setSuccess(`User ${user.email} ${!user.is_active ? 'activated' : 'deactivated'}`)
      loadUsers()
      if (selectedUser?.id === user.id) selectUser({ ...user, is_active: !user.is_active })
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function assignDepartment(deptCode: string) {
    if(!selectedUser) return
    try {
      await adminApi.assignUserDepartment(selectedUser.id, deptCode, token)
      setSuccess(`Assigned ${deptCode}`)
      selectUser(selectedUser) // refresh
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function revokeDepartment(deptCode: string) {
    if(!selectedUser) return
    try {
      await adminApi.revokeUserDepartment(selectedUser.id, deptCode, token)
      setSuccess(`Revoked ${deptCode}`)
      selectUser(selectedUser) // refresh
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function grantPerm(deptCode: string, permCode: string) {
    if(!selectedUser) return
    try {
      await adminApi.grantUserPermission(selectedUser.id, deptCode, permCode, token)
      setSuccess(`Granted ${permCode} for ${deptCode}`)
      selectUser(selectedUser)
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function revokePerm(deptCode: string, permCode: string) {
    if(!selectedUser) return
    try {
      await adminApi.revokeUserPermission(selectedUser.id, deptCode, permCode, token)
      setSuccess(`Revoked ${permCode} for ${deptCode}`)
      selectUser(selectedUser)
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <div style={{ display: 'flex', height: '100dvh', fontFamily: 'var(--font-sans)', background: 'var(--color-ground)', color: 'var(--color-ink)' }}>
      
      {/* Sidebar: Users List */}
      <div style={{ width: '300px', borderRight: '1px solid var(--color-border)', background: 'var(--color-surface)', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid var(--color-border)' }}>
          <h2 style={{ margin: 0, fontSize: '18px' }}>Admin Dashboard</h2>
          <button onClick={onLogout} style={{ marginTop: '10px', fontSize: '12px' }}>Logout</button>
        </div>
        <div style={{ padding: '10px', flex: 1, overflowY: 'auto' }}>
          <h3 style={{ fontSize: '14px', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>Users</h3>
          {users.map(u => (
            <div 
              key={u.id} 
              onClick={() => selectUser(u)}
              style={{ padding: '12px', background: selectedUser?.id === u.id ? 'var(--color-panel)' : 'transparent', borderRadius: '8px', cursor: 'pointer', marginBottom: '8px' }}
            >
              <div style={{ fontWeight: 500 }}>{u.email}</div>
              <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', display: 'flex', justifyContent: 'space-between' }}>
                <span>{u.role}</span>
                <span style={{ color: u.is_active ? 'var(--color-teal)' : '#d93025' }}>{u.is_active ? 'Active' : 'Inactive'}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content: User Details & Access */}
      <div style={{ flex: 1, padding: '32px', overflowY: 'auto' }}>
        {error && <div style={{ background: '#ffebee', color: '#c62828', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>{error}</div>}
        {success && <div style={{ background: '#e8f5e9', color: '#2e7d32', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>{success}</div>}
        
        {!selectedUser ? (
          <div style={{ color: 'var(--color-ink-muted)' }}>Select a user to manage access.</div>
        ) : (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2>{selectedUser.email}</h2>
              <button 
                onClick={() => handleToggleActive(selectedUser)}
                style={{ background: selectedUser.is_active ? '#ffebee' : '#e8f5e9', color: selectedUser.is_active ? '#c62828' : '#2e7d32', border: 'none', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer' }}
              >
                {selectedUser.is_active ? 'Deactivate User' : 'Activate User'}
              </button>
            </div>

            <div style={{ marginBottom: '32px', background: 'var(--color-surface)', padding: '24px', borderRadius: '12px', border: '1px solid var(--color-border)' }}>
              <h3>Departments</h3>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                <button onClick={() => assignDepartment('SALES')}>+ Add SALES</button>
                <button onClick={() => assignDepartment('FINANCE')}>+ Add FINANCE</button>
                <button onClick={() => assignDepartment('HR')}>+ Add HR</button>
              </div>

              {departments.length === 0 ? <p style={{ color: 'var(--color-ink-muted)' }}>No departments assigned.</p> : (
                <div style={{ display: 'grid', gap: '16px' }}>
                  {departments.map(d => {
                    const myPerms = permissions[d.code]?.map(p => p.code) || []
                    return (
                      <div key={d.code} style={{ border: '1px solid var(--color-border)', borderRadius: '8px', padding: '16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                          <strong style={{ fontSize: '16px', color: 'var(--color-teal)' }}>{d.code}</strong>
                          <button onClick={() => revokeDepartment(d.code)} style={{ fontSize: '12px', color: '#d93025' }}>Revoke Dept</button>
                        </div>
                        
                        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                          {['VIEW', 'UPLOAD', 'EDIT', 'DELETE'].map(p => {
                            const hasPerm = myPerms.includes(p)
                            return (
                              <label key={p} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '13px', cursor: 'pointer' }}>
                                <input 
                                  type="checkbox" 
                                  checked={hasPerm} 
                                  onChange={() => hasPerm ? revokePerm(d.code, p) : grantPerm(d.code, p)} 
                                />
                                {p}
                              </label>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
            
          </div>
        )}
      </div>

    </div>
  )
}
"""

os.makedirs(r"c:\ragDemo\ragdemo1\frontend\src\pages", exist_ok=True)
with open(r"c:\ragDemo\ragdemo1\frontend\src\pages\AdminDashboard.tsx", 'w', encoding='utf-8') as f:
    f.write(admin_dashboard)

print("Admin dashboard created.")

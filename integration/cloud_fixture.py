"""Cloud-test cleanup bound to an exact create response, never a prefix scan."""
from uuid import UUID


def archive_owned_project(client, *, name, project_id, run_id):
    """Archive only the synthetic project identified by this run's receipt.

    Some deployed servers omit project metadata from GetProjects. The exact
    server-assigned ID plus fresh UUID name from CreateProject is the ownership
    receipt; metadata, when present, is an additional consistency check.
    """
    # These are destructive-operation guards, not test assertions: they must
    # remain active under python -O / PYTHONOPTIMIZE as well.
    if UUID(run_id).hex != run_id:
        raise ValueError("Invalid synthetic run ID")
    if name != "memory-ci-" + run_id:
        raise ValueError("Not this run's synthetic project")
    if not project_id:
        raise ValueError("Missing CreateProject receipt")
    owned = client.get_project(name, include_deprecated=True)
    if owned is None:
        raise RuntimeError("Receipt project could not be verified")
    if owned.name != name or owned.project_id != project_id:
        raise RuntimeError("Project identity mismatch")
    marker = owned.metadata.get("memory_ci_owner")
    if marker is not None and marker != run_id:
        raise RuntimeError("Conflicting project ownership marker")
    if not owned.deprecated:
        result = client.delete_project(project_id, force=False)
        if not result.success:
            raise RuntimeError("Synthetic project cleanup failed")
    archived = client.get_project(name, include_deprecated=True)
    if archived is not None and not (
        archived.project_id == project_id and archived.name == name and archived.deprecated
    ):
        raise RuntimeError("Synthetic project was not archived")

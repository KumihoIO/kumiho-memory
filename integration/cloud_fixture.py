"""Cloud-test cleanup bound to an exact create response, never a prefix scan."""
from uuid import UUID


def archive_owned_project(client, *, name, project_id, run_id):
    """Archive only the synthetic project identified by this run's receipt.

    Some deployed servers omit project metadata from GetProjects. The exact
    server-assigned ID plus fresh UUID name from CreateProject is the ownership
    receipt; metadata, when present, is an additional consistency check.
    """
    assert UUID(run_id).hex == run_id, "Invalid synthetic run ID"
    assert name == "memory-ci-" + run_id, "Not this run's synthetic project"
    assert project_id, "Missing CreateProject receipt"
    owned = client.get_project(name, include_deprecated=True)
    assert owned is not None, "Receipt project could not be verified"
    assert owned.name == name and owned.project_id == project_id, "Project identity mismatch"
    marker = owned.metadata.get("memory_ci_owner")
    assert marker is None or marker == run_id, "Conflicting project ownership marker"
    if not owned.deprecated:
        result = client.delete_project(project_id, force=False)
        assert result.success, "Synthetic project cleanup failed"
    archived = client.get_project(name, include_deprecated=True)
    assert archived is None or (
        archived.project_id == project_id and archived.name == name and archived.deprecated
    ), "Synthetic project was not archived"

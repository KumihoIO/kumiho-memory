"""No SDK/network: exercise the exact-identity Cloud cleanup safety boundary."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from integration.cloud_fixture import archive_owned_project


RUN_ID = "a" * 32
NAME = "memory-ci-" + RUN_ID
PROJECT_ID = "server-assigned-project-id"


def project(**changes):
    fields = dict(name=NAME, project_id=PROJECT_ID, deprecated=False, metadata={})
    fields.update(changes)
    return SimpleNamespace(**fields)


def archive(client, **changes):
    receipt = dict(name=NAME, project_id=PROJECT_ID, run_id=RUN_ID)
    receipt.update(changes)
    archive_owned_project(client, **receipt)


@pytest.mark.parametrize("metadata", [{}, {"memory_ci_owner": RUN_ID}])
def test_exact_receipt_archives_with_or_without_project_metadata(metadata):
    client = Mock()
    client.get_project.side_effect = [project(metadata=metadata), project(deprecated=True)]
    client.delete_project.return_value = SimpleNamespace(success=True)
    archive(client)
    client.delete_project.assert_called_once_with(PROJECT_ID, force=False)
    assert client.get_project.call_count == 2


@pytest.mark.parametrize("changes", [
    {"name": "production"}, {"project_id": ""}, {"run_id": "not-a-uuid"},
])
def test_invalid_receipt_never_mutates(changes):
    client = Mock()
    with pytest.raises((AssertionError, ValueError)):
        archive(client, **changes)
    client.delete_project.assert_not_called()


@pytest.mark.parametrize("owned", [
    None, project(name="another-project"), project(project_id="another-id"),
    project(metadata={"memory_ci_owner": "another-run"}),
])
def test_unverified_or_conflicting_identity_never_mutates(owned):
    client = Mock()
    client.get_project.return_value = owned
    with pytest.raises(AssertionError):
        archive(client)
    client.delete_project.assert_not_called()


def test_archive_failure_is_not_reported_as_cleanup_success():
    client = Mock()
    client.get_project.return_value = project()
    client.delete_project.return_value = SimpleNamespace(success=False)
    with pytest.raises(AssertionError, match="cleanup failed"):
        archive(client)


@pytest.mark.parametrize("reread", [project(), project(project_id="another-id", deprecated=True)])
def test_archive_requires_fresh_server_confirmation(reread):
    client = Mock()
    client.get_project.side_effect = [project(), reread]
    client.delete_project.return_value = SimpleNamespace(success=True)
    with pytest.raises(AssertionError, match="not archived"):
        archive(client)


def test_recovery_of_already_archived_project_is_idempotent():
    client = Mock()
    client.get_project.return_value = project(deprecated=True)
    archive(client)
    client.delete_project.assert_not_called()

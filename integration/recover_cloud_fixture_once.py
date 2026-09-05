"""One-off receipt recovery, only on a dedicated test branch, never a sweep.

Evidence: https://github.com/KumihoIO/kumiho-memory/actions/runs/33945706066
Both live contracts passed; teardown recorded this exact ID/name and stopped
before archive because the Cloud response omitted project metadata.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from integration.cloud_fixture import archive_owned_project


def main():
    token = os.environ.get("KUMIHO_AUTH_TOKEN")
    assert token, "The existing Actions secret is required"
    run_id = "ba00911e6d3c4ccc8836422c9e1251a2"
    name = "memory-ci-" + run_id
    project_id = "c9a65b20-cbc6-4f38-a6bc-d1c812d59a09"
    with TemporaryDirectory(prefix="cloud-recovery-") as root:
        os.environ["KUMIHO_CONFIG_DIR"] = root
        os.environ["KUMIHO_CONTROL_PLANE_URL"] = "https://control.kumiho.cloud"
        import kumiho
        from kumiho.discovery import DiscoveryManager

        cache = Path(root) / "discovery.json"
        record = DiscoveryManager(
            control_plane_url="https://control.kumiho.cloud", cache_path=cache, timeout=15,
        ).resolve(id_token=token, force_refresh=True)
        target = record.region.grpc_authority or record.region.server_url
        host = urlparse(target if "://" in target else "https://" + target).hostname
        assert host and host.endswith(".kumiho.cloud"), "Not a Cloud endpoint"
        client = kumiho.client_from_discovery(
            id_token=token, control_plane_url="https://control.kumiho.cloud", cache_path=str(cache),
        )
        with kumiho.use_client(client):
            owned = client.get_project(name, include_deprecated=True)
            assert owned and owned.name == name and owned.project_id == project_id
            if not owned.deprecated:
                # Additional fixture-specific evidence, beyond the recorded
                # server ID/name. Never adopt or alter an unverified project.
                old = kumiho.get_revision(f"kref://{name}/prior.fact?r=1")
                dep = kumiho.get_revision(f"kref://{name}/grounded.decision?r=1")
                assert old.metadata.get("sentinel") == "keep"
                assert old.metadata.get("status") == "superseded"
                assert dep.metadata.get("grounding_stale") == "true"
                assert dep.metadata.get("grounding_stale_superseded_by") == f"kref://{name}/replacement.fact?r=1"
            archive_owned_project(client, name=name, project_id=project_id, run_id=run_id)
            print(f"Verified archived: {name} ({project_id}); force=False; receipt run=33945706066")


if __name__ == "__main__":
    main()

# Prepared publishing destination

No deployment is enabled. The native generator prepares complete private bundles
under `.data/publications/`. After approval, copy the reviewed bundle to
`www/project/` and review/enable `.github/workflows/project-page.yml.disabled`.
The workflow validates static artifacts and never starts training or data generation.

See [the publication guide](../docs/publication.md). No commit/push is performed
as part of preparation.

"""The GNSS-SDR images the pipeline runs, pinned by digest or image ID (D-04; containers/digests.md)."""

# stock v0.0.22 (D-04): the 108-byte tracking dump, one record per data symbol after bit sync (F23)
STOCK = "carlesfernandez/docker-gnsssdr:v0.0.22@sha256:befe06ed81163068e9eaf918024e18cedf1fc374925431320d57055650b48b79"

# containers/Dockerfile.patched on top of STOCK (D-08, D-27, D-40): 112-byte records with a float64 accumulated phase,
# one record per loop update, and GPS L5 codes with XB on its 8191-chip period (IS-GPS-705J 3.3.2.2). A locally built
# image, so it is pinned by its image ID, recorded in containers/digests.md. Image p2 since 19 Oct 2026; p1
# (sha256:1a63194c..., without 0003) stays pinned in E2 run 2's worktree.
PATCHED = "sha256:5bacd9f59966f09cb425897cb7828cec3721d75d592fdc2cec77840fc0c3694c"
PATCH_ID = "p2:0001-acc-phase-float64+0002-log-every-integration+0003-l5-xb-period"

# Offline zone foundations

The zone foundations are implemented for offline use and tests. They are not
connected to `FrameProcessor`, browser controls or the hardware controller.
`zone.enabled` remains rejected by configuration validation. No zone-following
or physical-registration qualification is claimed.

## Geometry

`app/zone_geometry.py` separates `ImagePoint` from `ReferencePoint`.
`ReferencePolygon` validates a simple polygon, including concave zones, within
the reference dimensions. Membership is inside, outside or uncertain: a point
whose declared uncertainty circle touches an edge is uncertain.

`ProjectiveMapping` accepts a caller-qualified transform and convex image
support. Its full input uncertainty rectangle must stay inside that support.
It rejects singular transforms and projective horizons. The output radius
covers transformed rectangle corners plus an explicitly declared reference error.
This is geometric propagation of a supplied bound, not statistical confidence
and not an error bound inferred from a feature-fitting residual.

## Reference files

`app/zone_reference.py` saves and reloads version-one metadata. `CameraIdentity`
binds the camera identifier, dimensions, supported orientation and calibration
hash. The reference binds a PNG image and its hash, registration surface, zone,
timezone-aware capture provenance and an explicitly `UnqualifiedPolicy`.

The registration surface must be convex; the contained zone may be concave.
Loading checks the complete schema, duplicate keys, image hash and decoded
dimensions, camera match and polygon containment. Stored image paths are relative
to the metadata file. Saving replaces metadata atomically and must not overwrite
the reference image. These checks make a coherent local artefact, not an accepted
physical zone.

## Registration candidates

`app/zone_registration.py` estimates a current-image-to-reference candidate from
ORB correspondences. Reference and current bird masks include a descriptor
footprint margin. Every fifth correspondence from the same match set is held out
of the RANSAC fit; this is not an independent validation dataset. Screening checks
inliers, held-out residuals, spatial coverage,
projected geometry, horizons, local distortion, camera bindings and freshness.

Each call estimates independently. Failure returns an invalid result; it does
not reuse an old transform. Candidates retain immutable coefficients, current
support, image hashes, capture times and diagnostic counts/residuals. They do not
construct a trusted `ProjectiveMapping` or supply a physical containment-error
allowance. The caller still needs independent registration qualification.

Pending work is listed in the
[verification and zone plan](independent-verification-and-zone-plan.md):
independent following verification, registration qualification, optical flow or
worker optimisation, reference drawing, configuration activation, selector and
controller integration, and physical planar-zone validation.

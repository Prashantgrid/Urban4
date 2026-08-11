-- Urban4 InfDB evidence contract (EPSG:4326 GeoJSON)
-- Pinned source revisions are recorded in INFDB_INTEGRATION.md.
-- Run the COPY statements with psql \copy so the client writes the CSV files.

\copy (
  SELECT objectid::text,
         ''::text AS osm_id,
         COALESCE(building_use_id::text, '') AS building_use_id,
         COALESCE(building_use::text, '') AS building_use,
         COALESCE(building_type::text, '') AS building_type,
         height, floor_area, floor_number, occupants, households,
         construction_year, postcode,
         concat_ws(' ', street, house_number) AS address,
         ST_AsGeoJSON(ST_Transform(geom, 4326)) AS geometry_json
  FROM basedata.buildings
  WHERE gemeindeschluessel = '09662000'
) TO 'infdb_buildings.csv' CSV HEADER;

\copy (
  SELECT id::text AS way_id,
         ST_AsGeoJSON(ST_Transform(geom, 4326)) AS geometry_json
  FROM basedata.ways_per_connection
  WHERE ags = '09662000'
) TO 'infdb_ways.csv' CSV HEADER;

\copy (
  SELECT b.objectid::text,
         b.assigned_way_id::text AS way_id,
         ST_AsGeoJSON(ST_Transform(cl.geom, 4326)) AS geometry_json
  FROM basedata.buildings b
  JOIN LATERAL (
      SELECT candidate.geom
      FROM basedata.connection_lines candidate
      WHERE candidate.ags = b.gemeindeschluessel
        AND candidate.connected_way_id = b.assigned_way_id
      ORDER BY candidate.geom <-> b.centroid
      LIMIT 1
  ) cl ON TRUE
  WHERE b.gemeindeschluessel = '09662000'
) TO 'infdb_connection_lines.csv' CSV HEADER;

\copy (
  SELECT version_id, objectid::text, grid_result_id,
         assigned_way_id::text AS way_id, peak_load_in_kw,
         vertice_id, connection_point
  FROM pylovo.buildings_result
  WHERE gemeindeschluessel = '09662000'
) TO 'pylovo_buildings.csv' CSV HEADER;

\copy (
  SELECT grid_result_id, version_id, kcid, bcid, plz,
         transformer_rated_power, transformer_equipment_name, model_status, power_flow_status,
         grid
  FROM pylovo.grid_result
  WHERE plz IN (SELECT DISTINCT postcode FROM basedata.buildings WHERE ags = '09662000')
) TO 'pylovo_grid_result.csv' CSV HEADER;

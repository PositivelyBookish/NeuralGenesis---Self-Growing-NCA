// Load Ahmedabad district boundary
var indiaDistricts = ee
  .FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.eq("ADM1_NAME", "Gujarat"))
  .filter(ee.Filter.eq("ADM2_NAME", "Ahmadabad"));

Map.centerObject(indiaDistricts, 9);
Map.addLayer(indiaDistricts, { color: "red" }, "Ahmedabad Boundary");

// Load Landsat 8 data for 2015
var startDate = ee.Date.fromYMD(2015, 1, 1);
var endDate = ee.Date.fromYMD(2015, 12, 31);

var landsat = ee
  .ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(indiaDistricts)
  .filterDate(startDate, endDate)
  .sort("CLOUD_COVER")
  .median()
  .clip(indiaDistricts);

// Cloud masking function
function cloudMaskOli(image) {
  var qa = image.select("QA_PIXEL").toInt();
  var mask = qa
    .bitwiseAnd(1 << 1)
    .eq(0)
    .and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0));

  return image
    .select(
      ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
      ["B2", "B3", "B4", "B5", "B6", "B7"]
    )
    .updateMask(mask);
}

var maskedImage = cloudMaskOli(landsat);

// Compute NDBI and MNDWI
var bandMap = {
  NIR: maskedImage.select("B5"),
  SWIR: maskedImage.select("B6"),
  GREEN: maskedImage.select("B3"),
};

var ndbi = maskedImage
  .expression("(SWIR - NIR) / (SWIR + NIR)", bandMap)
  .rename("NDBI");
Map.addLayer(
  ndbi,
  { min: -1, max: 1, palette: ["blue", "white", "red"] },
  "NDBI (Built-up)",
  false
);

var mndwi = maskedImage
  .expression("(GREEN - SWIR) / (GREEN + SWIR)", bandMap)
  .rename("MNDWI");
Map.addLayer(
  mndwi,
  { min: -1, max: 1, palette: ["red", "white", "blue"] },
  "MNDWI (Water)",
  false
);

// Extract built-up areas
var builtUp = ee
  .Image(0)
  .where(ndbi.gt(-0.1).and(mndwi.lte(0)), 1)
  .selfMask()
  .clip(indiaDistricts);
Map.addLayer(builtUp, { palette: "red" }, "Built-up Area (2015)", true);

// Export built-up area map
Export.image.toDrive({
  image: builtUp,
  description: "Ahmedabad_BuiltUp_2015",
  folder: "EarthEngineData",
  scale: 30,
  region: indiaDistricts.geometry(),
  fileFormat: "GeoTIFF",
});

print("Exporting Built-up Area Map for Ahmedabad (2015)...");

# Adaptív multimodális szenzorfúzió autonóm járművek hibrid környezetmodelljében.

A repozitórium a 2026-os Tudományos Diákköri kutatásnak a C++ implementációját és a Python szkriptjeit tartalmazza. A modul a SZEnergy Team SZEmission járművének környezetérzékeléséhez készült, valós időben fuzionálva a szegmentált kameraképeket és a LiDAR pontfelhőket egy metrikus foglaltsági rácson.

*Főbb funkciók:* Inverz perspektivikus leképezés, adaptív paraméterezésű Log-Odds frissítés, valamint GPU-gyorsított Gauss-féle potenciálmező generálás.

*Adatok és reprodukálhatóság:* A nyers mérési adatok (.mcap) a csapat belső tesztjeit tartalmazzák, így ezek nem publikusak. Az ábrák generálásához és az eredmények reprodukálásához viszont minden feldolgozott .csv fájlt feltöltöttem, így a Python szkriptek ezekből önállóan is futtathatók.

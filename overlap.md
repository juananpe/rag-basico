El `chunk_overlap` (solapamiento entre fragmentos) sirve para evitar que se pierda contexto cuando una idea importante queda partida justo en el borde entre dos chunks. Un ejemplo concreto lo deja muy claro:

**Sin overlap**

Imagina que estás indexando un manual de políticas de una empresa, con chunks de 200 tokens y overlap = 0. El documento contiene este pasaje:

> "...Los empleados con más de 5 años de antigüedad tienen derecho a 30 días de vacaciones. **[corte de chunk aquí]** Este beneficio no aplica a contratistas externos ni a empleados en periodo de prueba..."

El corte cae justo en medio de la idea. Resultado:

- **Chunk A** termina con: "...tienen derecho a 30 días de vacaciones."
- **Chunk B** empieza con: "Este beneficio no aplica a contratistas externos..."

Si un usuario pregunta *"¿Los contratistas tienen 30 días de vacaciones?"*, el retriever probablemente recupere el Chunk A (que menciona "30 días de vacaciones") pero no el Chunk B, porque el Chunk B por sí solo es ambiguo: "este beneficio" no tiene referente claro y su embedding no se parece mucho a la pregunta. El LLM podría responder erróneamente que **sí**, porque solo ve el Chunk A.

**Con overlap (por ejemplo, 50 tokens)**

El Chunk B ahora empieza unos tokens antes:

- **Chunk B** contiene: "...tienen derecho a 30 días de vacaciones. Este beneficio no aplica a contratistas externos..."

Ahora ese chunk es autocontenido: la excepción aparece junto a la regla. Su embedding sí se parece a la pregunta del usuario (menciona vacaciones, 30 días y contratistas a la vez), así que se recupera correctamente y el LLM responde que **no aplica a contratistas**.

**La intuición general**

El overlap actúa como un "seguro" contra cortes arbitrarios: como no controlas dónde caen los límites de los chunks respecto a la estructura semántica del texto, duplicar un poco de contenido en las fronteras garantiza que cualquier idea que cruce un límite exista completa en al menos un chunk. El coste es algo de redundancia en el índice (más almacenamiento y chunks ligeramente repetidos), por eso valores típicos rondan el 10–20% del tamaño del chunk.

Vale la pena notar que el overlap es una solución barata pero "tonta"; alternativas más sofisticadas como el chunking semántico (cortar por párrafos, secciones o cambios de tema) atacan el mismo problema de raíz, aunque son más costosas de implementar.
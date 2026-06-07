1. Para el modelo, tenemos varias opciones, como "deepseek/deepseek-v4-flash", "openai/o4-mini", "google/gemini-3.1-flash-lite", entre otros. En este ejemplo, usaremos "deepseek/deepseek-v4-flash". ¿Por qué elegimos este modelo? ¿Qué tamaño de contexto tiene? ¿Qué precio tiene?  ¿Qué latencia tiene? 

2. Prueba a ingerir un documento con información en Euskera. ¿Qué modelo se comporta mejor?

3. Prueba el pdf_loader con un pdf que tenga tablas. ¿Cómo se comporta? ¿Qué modelo se comporta mejor?

4. Queremos mantener las URLs de noticias actualizadas (cuando el ayuntamiento de Donostia publique nuevas noticias, queremos que se añadan a nuestra base de datos vectorial). ¿Cómo lo harías?

5. Con el comando:

    ```
    python test_load_pdf.py  ./cas-plan-donostia-gazteria-2025-2027.pdf
    ```

   El script test_load_pdf.py carga el PDF y lo divide en fragmentos. ¿Cuántos fragmentos se crean? ¿Cuál es el tamaño de cada fragmento? ¿Cómo se determina el tamaño de los fragmentos? 

6. Basándote en el script `bare_minimum.py`, modifica test_load_pdf.py para que guarde los documentos en una base de datos vectorial ChromaDB (en la colección plan_donostia_gazteria) utilizando el modelo "deepseek/deepseek-v4-flash". Prueba a lanzar la consulta "¿Cuál es la misión del plan Gazteria?" para comprobar si el modelo responde correctamente.

6.1 Prueba a cambiar el modelo de embeddings a los siguientes:

intfloat/multilingual-e5-large	
openai/text-embedding-3-large	
openai/text-embedding-3-small
google/gemini-embedding-2	

 ¿Cómo afecta esto a los resultados de la consulta? ¿Qué modelo se comporta mejor? ¿cuál es el más barato? ¿cuál es el más rápido? 

6.2 Según la doc de OpenRouter, google/gemini-embedding-2 acepta dimensiones desde 128 hasta 3072, con las recomendadas siendo 768, 1536, o 3072.

Usa esta orden curl para comprobar cuál es la dimensión por defecto de los embeddings de  google/gemini-embedding-2:

```
curl -s "https://openrouter.ai/api/v1/embeddings" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"google/gemini-embedding-2","input":"Hola"}' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('data[0] keys:', list(data['data'][0].keys()))
print('embedding len:', len(data['data'][0]['embedding']))
print('usage:', data['usage'])
"
````

7. El fichero PDF Plan DONOSTIA GAZTERIA 2025-2027 (cas-plan-donostia-gazteria-2025-2027.pdf) proviene de https://www.donostia.eus/documents/d/asset-library-100431/cas-plan-donostia-gazteria-2025-2027. Sin embargo, el script de ingestión pdf_loader.py no guarda la URL de origen del documento. ¿Cómo modificarías el script para que guarde esta información?

8. Now, let's ingest dk-bases-especificas-2026-v22-01-2026.pdf 
https://www.donostia.eus/documents/d/asset-library-802119/dk-bases-especificas-2026-v22-01-2026
into another collection (the pdf is about "Subvenciones a entidades sin ánimo de lucro 2026") It is a 2-column pdf, one column for basque and the other for spanish so I think that it will be a hard nut to crack for some embedding models. I want to test it with the question: ¿cuáles son los criterios de valoración?

9. Queremos crear un chatbot para responder preguntas sobre el plan DONOSTIA GAZTERIA 2025-2027 utilizando el modelo "deepseek/deepseek-v4-flash". 




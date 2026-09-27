(() => {
  const replacements = [
    ["Contrastive Learning con pocas etiquetas", "Contrastive Learning with Few Labels"],
    ["¿Puede SimCLR aprender representaciones útiles de MNIST sin conocer las clases?", "Can SimCLR learn useful MNIST representations without knowing the classes?"],
    ["Presentación reproducible", "Reproducible presentation"],
    ["Resultados cargados automáticamente", "Automatically loaded results"],
    ["Sin cifras inventadas", "No fabricated numbers"],
    ["Escanea para abrir la demo", "Scan to open the demo"],
    ["Conclusiones e interpretación de los resultados", "Conclusions and interpretation of results"],
    ["El control que aísla el efecto contrastivo", "The control that isolates the contrastive effect"],
    ["Evidencia dentro del espacio de embeddings", "Evidence inside the embedding space"],
    ["¿Qué imágenes considera parecidas?", "Which images does the model consider similar?"],
    ["¿Qué aportó realmente SimCLR?", "What did SimCLR actually contribute?"],
    ["Resultados de clasificación", "Classification results"],
    ["Diseño experimental", "Experimental design"],
    ["¿Qué aprende SimCLR?", "What does SimCLR learn?"],
    ["Comparación global", "Overall comparison"],
    ["El problema", "The problem"],
    ["Limitaciones", "Limitations"],
    ["Conclusión respaldada por los datos", "Data-supported conclusion"],
    ["Los resultados son internamente coherentes", "The results are internally consistent"],
    ["SimCLR sí aprendió una representación útil", "SimCLR did learn a useful representation"],
    ["¿Por qué Logistic Regression gana con pocas etiquetas?", "Why does Logistic Regression win with few labels?"],
    ["El calendario de entrenamiento es limitado", "The training schedule is limited"],
    ["El preentrenamiento contrastivo cambia claramente el resultado", "Contrastive pretraining clearly changes the result"],
    ["Con pocas etiquetas, SimCLR ya es competitivo", "With few labels, SimCLR is already competitive"],
    ["Con todas las etiquetas cambia el segundo puesto", "With all labels, second place changes"],
    ["Qué podemos concluir, y qué no", "What we can and cannot conclude"],
    ["La arquitectura también influye", "The architecture also matters"],
    ["Accuracy y F1 cuentan una historia compatible", "Accuracy and F1 tell a consistent story"],
    ["Pregunta de investigación", "Research question"],
    ["Etiquetar datos cuesta tiempo y dinero, pero conseguir imágenes sin etiquetar suele ser más fácil.", "Labelling data costs time and money, while obtaining unlabelled images is usually easier."],
    ["Si SimCLR observa muchas imágenes ", "If SimCLR observes many images "],
    ["sin usar sus etiquetas", "without using their labels"],
    [", ¿necesitará después menos ejemplos etiquetados para reconocer los dígitos?", ", will it later need fewer labelled examples to recognise the digits?"],
    ["Comparamos tres disponibilidades de etiquetas:", "We compare three label-availability settings:"],
    ["Todas las imágenes de entrenamiento", "All training images"],
    ["CNN supervisada", "Supervised CNN"],
    ["SimCLR autosupervisado", "Self-supervised SimCLR"],
    ["sin etiquetas", "without labels"],
    ["encoder congelado", "frozen encoder"],
    ["mismo subconjunto etiquetado", "same labelled subset"],
    ["evaluación final en test", "final test evaluation"],
    ["También usamos ", "We also use "],
    [" como referencia sencilla y un ", " as a simple reference and a "],
    ["encoder aleatorio", "random encoder"],
    [" como control.", " as a control."],
    ["Ejecución mostrada:", "Displayed run:"],
    ["Semillas:", "Seeds:"],
    ["épocas SimCLR/supervisado/probe:", "SimCLR/supervised/probe epochs:"],
    ["Las particiones son estratificadas y no se solapan.", "The splits are stratified and do not overlap."],
    ["Para cada imagen generamos dos vistas moderadamente distintas:", "For each image, we generate two moderately different views:"],
    ["pequeñas rotaciones y traslaciones;", "small rotations and translations;"],
    ["cambios suaves de escala;", "mild scale changes;"],
    ["ruido ligero;", "light noise;"],
    ["sin flips", "no flips"],
    [", porque podrían alterar el significado de un dígito.", ", because they could change a digit's identity."],
    ["Las dos vistas de la misma imagen forman un ", "The two views of the same image form a "],
    ["par positivo", "positive pair"],
    [". Las demás vistas del batch son ", ". The other views in the batch are "],
    ["negativos", "negatives"],
    ["NT-Xent aproxima los positivos y separa los negativos. Las etiquetas nunca entran en esta pérdida.", "NT-Xent pulls positives together and pushes negatives apart. Labels never enter this loss."],
    ["Comparamos dos encoders con ", "We compare two encoders with "],
    ["la misma arquitectura", "the same architecture"],
    ["Clasificador", "Classifier"],
    ["Etiquetas", "Labels"],
    ["Sin preentrenamiento", "No pretraining"],
    ["Pesos aleatorios y congelados", "Random frozen weights"],
    ["Lineal", "Linear"],
    ["Preentrenado sin etiquetas y congelado", "Pretrained without labels and frozen"],
    ["El mismo lineal", "The same linear classifier"],
    ["Los mismos IDs", "The same IDs"],
    ["Interpretación", "Interpretation"],
    ["Si el linear probe de SimCLR supera al probe aleatorio, la mejora procede de la representación aprendida, no de un clasificador más potente.", "If the SimCLR linear probe beats the random probe, the gain comes from the learned representation, not from a more powerful classifier."],
    ["Encoder aleatorio + probe", "Random encoder + probe"],
    ["Regresión logística", "Logistic Regression"],
    ["Leyenda del semáforo:", "Traffic-light legend:"],
    ["Aviso:", "Warning:"],
    ["se usa el dataset completo, pero con una sola semilla y calendario reducido (8/10/10 épocas) para limitar el coste de cómputo. No equivale al protocolo final de tres semillas y 30 épocas.", "the full dataset is used, but with one seed and a reduced schedule (8/10/10 epochs) to limit compute cost. This is not equivalent to the final three-seed, 30-epoch protocol."],
    ["aún no está completo; se muestran resultados", "is not complete yet; results from"],
    ["Al terminar la ejecución reducida y volver a ejecutar este notebook, cambiará automáticamente a", "are shown. After the reduced run finishes and this notebook is executed again, it will automatically switch to"],
    ["estos son resultados", "these are"],
    ["con una semilla; validan el pipeline, pero no constituyen todavía conclusiones generales.", "results with one seed; they validate the pipeline but do not yet support general conclusions."],
    ["Ganancia del encoder preentrenado frente al aleatorio:", "Gain of the pretrained encoder over the random encoder:"],
    ["puntos", "points"],
    ["Separación media positivo–negativo:", "Mean positive–negative separation:"],
    ["antes y", "before and"],
    ["después de SimCLR.", "after SimCLR."],
    ["Los vecinos se buscan solo mediante los embeddings. Las clases se muestran después, exclusivamente para interpretar la representación.", "Neighbours are retrieved using only the embeddings. Class labels are shown afterwards solely to interpret the representation."],
    ["Lectura correcta:", "Correct interpretation:"],
    ["SimCLR debe compararse primero con el encoder aleatorio para medir la calidad aprendida. Compararlo con la CNN y Logistic Regression responde a una pregunta adicional: si esa representación produce el mejor clasificador final.", "SimCLR should first be compared with the random encoder to measure what the representation learned. Comparing it with the CNN and Logistic Regression answers an additional question: whether that representation yields the best final classifier."],
    ["MNIST es pequeño, centrado y mucho más sencillo que imágenes naturales.", "MNIST is small, centred, and much simpler than natural images."],
    ["SimCLR es sensible al tamaño del batch y al número de épocas.", "SimCLR is sensitive to batch size and the number of epochs."],
    ["El método contrastivo ve más imágenes sin etiquetar y consume más cómputo: no es una comparación de coste equivalente.", "The contrastive method sees additional unlabelled images and uses more compute: this is not a compute-matched comparison."],
    ["El linear probe mide separabilidad lineal, no todo lo que podría obtenerse mediante fine-tuning.", "The linear probe measures linear separability, not everything that fine-tuning might recover."],
    ["La ejecución mostrada usa una sola semilla; el protocolo final requiere tres semillas y 30 épocas por fase.", "The displayed run uses one seed; the final protocol requires three seeds and 30 epochs per stage."],
    ["usa todos los datos, pero un calendario reducido de 8/10/10 épocas para limitar el coste de cómputo.", "uses all data, but a reduced 8/10/10-epoch schedule to limit compute cost."],
    ["Una ejecución", "A"],
    ["no permite conclusiones generales; el protocolo final requiere tres semillas.", "run cannot support general conclusions; the final protocol requires three seeds."],
    ["Las etiquetas de test se utilizan una vez fijado el modelo, nunca para seleccionar hiperparámetros.", "Test labels are used only after fixing the model, never to select hyperparameters."],
    ["Al aumentar las etiquetas, todos los modelos basados en la CNN mejoran:", "As label availability increases, all CNN-based models improve:"],
    ["la CNN supervisada pasa de", "the supervised CNN rises from"],
    ["SimCLR + linear probe pasa de", "SimCLR + linear probe rises from"],
    ["Esta tendencia monotónica indica que los subconjuntos etiquetados y la evaluación se comportan de forma razonable.", "This monotonic trend indicates that the labelled subsets and evaluation behave as expected."],
    ["Frente al mismo encoder aleatorio y congelado, SimCLR mejora", "Against the same frozen random encoder, SimCLR improves by"],
    ["La ganancia media es", "The mean gain is"],
    ["puntos. Además, la loss contrastiva baja de", "points. In addition, contrastive loss falls from"],
    ["Las tres medidas respaldan que SimCLR aprendió estructura útil y que la mejora no procede únicamente del clasificador lineal.", "All three measures support that SimCLR learned useful structure and that the improvement does not come solely from the linear classifier."],
    ["Con solo 480 etiquetas (1 %), SimCLR alcanza", "With only 480 labels (1%), SimCLR reaches"],
    ["sin preentrenamiento,", "without pretraining,"],
    ["de Logistic Regression y", "for Logistic Regression and"],
    ["de la CNN supervisada. Con 10 % obtiene", "for the supervised CNN. With 10% it reaches"],
    ["Esto demuestra utilidad con pocas etiquetas en esta ejecución, aunque no superioridad: queda", "This demonstrates usefulness with few labels in this run, but not superiority: it remains"],
    ["por debajo de la CNN en 1 % y 10 %.", "below the CNN at 1% and 10%."],
    ["Con 100 % de etiquetas, la CNN supervisada sigue siendo la mejor con", "With 100% of the labels, the supervised CNN remains best at"],
    ["SimCLR llega a", "SimCLR reaches"],
    ["supera a Logistic Regression en", "it exceeds Logistic Regression by"],
    ["pero queda", "but remains"],
    ["por debajo de la CNN. Por tanto, el encoder congelado es muy informativo, aunque el entrenamiento end-to-end todavía aprovecha mejor las etiquetas.", "below the CNN. The frozen encoder is therefore highly informative, although end-to-end training still makes better use of the labels."],
    ["con 1 %, 10 % y 100 % de etiquetas. La mejora media correcta es", "with 1%, 10%, and 100% labels. The correct mean gain is"],
    [", no miles de puntos. Además, la loss contrastiva baja de", ", not thousands of points. In addition, contrastive loss falls from"],
    ["y la separación coseno positivo–negativo aumenta de", "and positive–negative cosine separation increases from"],
    ["Las tres evidencias apuntan en la misma dirección: el preentrenamiento no es equivalente a dejar la red aleatoria.", "All three pieces of evidence agree: pretraining is not equivalent to leaving the network random."],
    ["Obtiene", "It obtains"],
    ["Esto es plausible en MNIST: las imágenes están centradas, alineadas, tienen fondo uniforme y la posición de cada píxel es directamente informativa.", "This is plausible on MNIST: images are centred and aligned, backgrounds are uniform, and each pixel's position is directly informative."],
    ["Un modelo lineal tiene pocos parámetros y puede ser muy eficiente con pocos ejemplos. No demuestra que sea superior en problemas visuales más complejos.", "A linear model has few parameters and can be highly sample-efficient. This does not show that it is superior on more complex visual problems."],
    ["SimCLR recibe", "SimCLR receives"],
    ["de preentrenamiento y la CNN supervisada", "of pretraining, while the supervised CNN receives"],
    ["épocas como máximo", "epochs at most"],
    ["Esta ejecución está condicionada por el presupuesto temporal y no representa necesariamente la capacidad final de los modelos.", "This run is constrained by the time budget and does not necessarily represent the models' final capacity."],
    ["épocas", "epochs"],
    ["de preentrenamiento y la CNN supervisada parte de cero con un calendario muy corto.", "of pretraining, while the supervised CNN starts from scratch with a very short schedule."],
    ["de la CNN con 1.200 etiquetas no representa su capacidad final: su validation accuracy seguía aumentando en la última época.", "for the CNN with 1,200 labels does not represent its final capacity: validation accuracy was still increasing in the last epoch."],
    ["SimCLR supera ampliamente al encoder aleatorio, pero dos épocas no bastan para esperar una representación madura.", "SimCLR clearly beats the random encoder, but two epochs are not enough to expect a mature representation."],
    ["El encoder termina con", "The encoder ends with"],
    [", que comprime la distribución espacial en un vector global.", ", which compresses the spatial distribution into a global vector."],
    ["Logistic Regression conserva los 784 píxeles y sus posiciones.", "Logistic Regression preserves all 784 pixels and their positions."],
    ["Esto puede favorecer al baseline lineal en MNIST, aunque la comparación principal SimCLR–aleatorio sigue siendo justa porque ambos utilizan exactamente el mismo encoder.", "This can favour the linear baseline on MNIST, although the main SimCLR–random comparison remains fair because both use exactly the same encoder."],
    ["En los modelos débiles, F1 macro es menor que accuracy porque las matrices de confusión muestran que algunas clases se predicen mucho mejor que otras.", "For weak models, macro F1 is below accuracy because the confusion matrices show that some classes are predicted much better than others."],
    ["El encoder aleatorio, por ejemplo, alcanza", "The random encoder, for example, reaches"],
    ["de accuracy con todas las etiquetas, pero su F1 macro es mucho menor.", "accuracy with all labels, but its macro F1 is much lower."],
    ["Esto es compatible con un clasificador que se concentra en unas pocas clases; no parece un fallo del cálculo de métricas.", "This is consistent with a classifier focusing on a few classes; it does not look like a metric-computation error."],
    ["En la ejecución", "In run"],
    [", SimCLR aprende sin etiquetas una representación fuertemente separable y alcanza", ", SimCLR learns a strongly separable representation without labels and reaches"],
    ["usando solo el 1 % de las etiquetas. Esto respalda que el preentrenamiento contrastivo puede producir representaciones útiles con poca supervisión.", "using only 1% of the labels. This supports that contrastive pretraining can produce useful representations with little supervision."],
    ["Sin embargo, la CNN supervisada obtiene la mayor accuracy en los tres presupuestos.", "However, the supervised CNN achieves the highest accuracy at all three budgets."],
    ["No podemos afirmar que SimCLR sea el mejor método ni generalizar el resultado a otras semillas o datasets.", "We cannot claim that SimCLR is the best method or generalise the result to other seeds or datasets."],
    ["Esta ejecución usa una sola semilla y un calendario reducido de", "This run uses one seed and a reduced schedule of"],
    ["el protocolo confirmatorio sigue requiriendo tres semillas.", "the confirmatory protocol still requires three seeds."],
    [", SimCLR aprendió sin etiquetas una representación más útil que la de un encoder aleatorio. La comparación describe esta ejecución concreta, pero", ", SimCLR learned a more useful representation than a random encoder without labels. This comparison describes this specific run, but it"],
    [", SimCLR aprendió sin etiquetas una representación claramente más útil que la de un encoder aleatorio.", ", SimCLR learned a clearly more useful representation than a random encoder without using labels."],
    ["Sin embargo, no superó a la CNN supervisada ni a Logistic Regression.", "However, it did not outperform the supervised CNN or Logistic Regression."],
    ["Por tanto, esta demo demuestra que el pipeline contrastivo funciona, pero", "Therefore, this demo shows that the contrastive pipeline works, but"],
    ["no demuestra que SimCLR reduzca la necesidad de etiquetas en general", "does not show that SimCLR generally reduces the need for labels"],
    ["No debemos afirmar que", "We should not claim that"],
    ["SimCLR funciona mejor con pocas etiquetas", "SimCLR works better with few labels"],
    ["basándonos en una sola semilla", "based on a single seed"],
    [": con 1 % obtiene", ": with 1% it obtains"],
    [", frente a", ", compared with"],
    ["de la CNN y", "for the CNN and"],
    ["de Logistic Regression. Para responder la hipótesis principal con rigor harían falta tres semillas y el calendario completo de 30 épocas.", "for Logistic Regression. Answering the main hypothesis rigorously would require three seeds and the complete 30-epoch schedule."],
    ["Gracias", "Thank you"],
    ["Preguntas", "Questions"],
    ["Idea clave:", "Key idea:"],
    ["el éxito contrastivo se mide primero por cuánto mejora la representación frente al mismo encoder sin preentrenar, no por asumir que debe ganar siempre a todos los clasificadores.", "contrastive success is first measured by how much the representation improves over the same untrained encoder, not by assuming it must always beat every classifier."],
    ["Entrenamiento", "Training"],
    ["Validación", "Validation"]
  ].sort((a, b) => b[0].length - a[0].length);

  function translate(text) {
    return replacements.reduce((value, [es, en]) => value.split(es).join(en), text);
  }

  function translateNode(node, text) {
    let value = translate(text);
    if (text.trim() === 'y') value = text.replace('y', 'and');
    const cell = node.parentElement && node.parentElement.closest('.jp-Cell');
    if (cell && cell.id === 'cell-id=conclusions') {
      if (text.trim() === 'a') value = text.replace('a', 'to');
      value = value.replace(/\bEl\s*$/, 'The ');
    }
    return value;
  }

  function initialise() {
    const header = document.querySelector("main > .jp-MarkdownCell[id='cell-id=title']");
    const main = document.querySelector("main");
    if (!header || !main || header.querySelector('.language-switcher')) return;

    const switcher = document.createElement('div');
    switcher.className = 'language-switcher';
    switcher.setAttribute('role', 'group');
    switcher.setAttribute('aria-label', 'Language / Idioma');
    switcher.innerHTML = '<button type="button" data-lang="en" aria-pressed="true">EN</button><button type="button" data-lang="es" aria-pressed="false">ES</button>';
    header.appendChild(switcher);

    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const parent = node.parentElement;
        if (!parent || parent.closest('style, script, .language-switcher')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      nodes.push({ node, es: node.nodeValue, en: translateNode(node, node.nodeValue) });
    }

    function setLanguage(language) {
      nodes.forEach(item => { item.node.nodeValue = item[language]; });
      document.documentElement.lang = language;
      document.title = language === 'en'
        ? 'Contrastive Learning with Few Labels'
        : 'Contrastive Learning con pocas etiquetas';
      switcher.querySelectorAll('button').forEach(button => {
        const active = button.dataset.lang === language;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
    }

    switcher.addEventListener('click', event => {
      const button = event.target.closest('button[data-lang]');
      if (button) {
        setLanguage(button.dataset.lang);
        const url = new URL(window.location.href);
        if (button.dataset.lang === 'es') url.searchParams.set('lang', 'es');
        else url.searchParams.delete('lang');
        window.history.replaceState({}, '', url);
      }
    });
    const requestedLanguage = new URLSearchParams(window.location.search).get('lang');
    setLanguage(requestedLanguage === 'es' ? 'es' : 'en');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialise);
  } else {
    initialise();
  }
})();

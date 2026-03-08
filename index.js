import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    
    const model = tf.sequential()

    /**
     * Primeira camada da rede:
     * Entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)
     */

    /**
     * 80 neuronios -> Tudo isso porque tem pouca base de treino
     * Quanto mais neuronios maior é a complexibilidade a rede pode aprender
     * Consequentemente mais processamento ela vai usar
     */

    /**
     * A ReLU age como um filtro:
     * É como se ela deixasse somente os dados uteis seguirem viagem na rede
     * Se a info for positiva passa para frente
     * Se for zero ou negativa, pode jogar fora.
     */

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    /**
     * Saída: 3 neuronios 
     * 1 para cada categoria (premium, medium, basic)
     */

    /**
     * Activation: softmax normaliza a saida em probabilidades
     */

    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    /**
     * Compilando o modelo
     * optimizer Adam ( Adaptive Moment Estimation)
     * É um treinador pessoal moderno para redes neurais:
     * Ajusta os pesos de forma eficiente e inteligente
     * Aprender com historico de erros e acertos
     */

    /**
     * loss: categoricalCrossentropy
     * Ele compara o que o modelo "acha" (os scores de cada categoria)
     * com a resposta certa.
     * A categoria premium sera sempre [1,0,0]
     */

    /**
     * Quanto mais distante da previsão do modelo da resposta correta
     * maior o erro (loss)
     */

    // Exemplo classico: classificaçào de imagens, recomendação, categorização de usuário
    // Qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy', 
        metrics: ['accuracy']
    })

    /**
     * Treinamento do modelo
     */

    await model.fit(
        inputXs,
        outputYs,
        { 
            verbose: 0, // desabilita logs internos
            epochs: 100, // Rodar na nossa base de dados (dataset) 100x
            shuffle: true, // Embaralhar os dados para evitar viés
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // ) 
            }
        }
    )

    return model

}

async function predict(model, pessoa) {

    /**
     * Transformar o array js para o tensor (tfjs)
     */

    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    
    return predArray[0].map((prob, index) => ({prob, index}))
    
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [ 
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


/**
 * Quanto mais dado, melhor!
 * Assim o algoritmo consegue entender melhor os padrões complexos dos dados
 */
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28-25)/(40-25) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2,// idade normalizada
        1, // azul
        0, // vermelho
        0, // verde
        0, // são paulo
        1, // rio
        0, // curitiba
    ]
]

const predctions = await predict(model, pessoaTensorNormalizado) 

const result = predctions
    .sort((a,b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob*100).toFixed(2)}%)`)
    .join('\n')


console.log(result)
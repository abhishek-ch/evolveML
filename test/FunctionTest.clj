 (ns FunctionTest)
(println "Test Function")

(defn testMap
 [input]
 (str input " Hi hru?"))

(println "Seq can be treated as Seq " (map testMap ["Abhishek " "Buntha"]))
(println "Seq can be treated as Seq " (map testMap ))
(println "Exact map use-case "
       (into {} (map (fn
                   [[key value]]
                   [key (inc value)])
                  {:name 1 :value 1}))
       )

(println "Convert map to Sequence "(map identity {:name "Abhishek" :value "Buntha"}))
(println "map inside collection " (map str ["a" "b" "c"] ["d" ["e" "f"] "g" "h"] ))

(def sum
 #(reduce + %))

(def divide
 #(/ (sum %) (count %) )
 )

(defn stats
 [numbers]
 (map #(% numbers) [sum count divide])
 )

(println "map as a function 1 " (stats [1 2 3 4 5]))
(println "map as a function 2 " (stats [10 3 40]))
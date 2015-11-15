;; CLOJURE CHEAT SHEET http://clojure.org/cheatsheet
(ns Wonder)

  (conj '(1 2 3) 4)

 (println "Hello World")
 (println (+ 2 3))

 (when true
  (println "First")
  (println "Second")
  "going ahead")


 (def check_def_vector
  ["abh","Chou","OK"])

 (println check_def_vector)

(println "is it nil" (nil? 1))
 (println "is nill euqal " (= nil nil))
(println (= 1 1))                                           ;true
 (println (= "abhi" "abhi"))                                ;true
(println (= "abhi" "ABHI"))                                 ; false

 (println (/ 1 5))
 (println (+ 2.3 4))
(println (/ 1.0 5))

 (def value "Abhishek")
 (println (str value " choudhary " ))

 (def map_val {:a 1
               :b "abhsh"
               :c [1 2 3]
               :e {:f "f inside"}}
               )

(println "map as function" (map_val :a) (map_val :d) )
(println "a map key" (map_val :c) )
(println (get map_val :a))
(println "get - in" (get-in map_val [:e :f]))
(println "Default value " ( :d {:a 1 :b 2 :c 3} "FULLTU"))
(println "Hash Map" (get (hash-map :a 1 :b 2 :c 3) :a) )
(println "Set with no value available" (get #{:a :b :c} :d))


 (println (first check_def_vector))
 (println 'check_def_vector)
 (println "Quoting " (second '(check_def_vector map_val 0 1)))


;function
 (println ((and + - ) 1 2 3) ((or + - ) 1 2 3) )
 (println ((first [+ 0])  1 2 3))
 (println (+ (inc 199) (/ 100 (- 7 2))))

 (if false
          (println "Abhshek")
          (println "OK"))


;function

 (defn testing_function
  "ok this is ignorabe doc string"
  [name read]
  (str "OK U R GOOD \"" name "\" BUT U R READING \"" read "\" THATS GREAT"))
 (println (testing_function "Abhishek" "clojure"))


 (defn func_declare
  [x]
  (if (> x 6)
   "Thats awesome .."
   "You just lacked a bit")
 )



 (println (func_declare 7))
(println (func_declare 5))

;anonymous fnction
 (println (map (fn [name] (str " Hi... " name))
             ["Abhsh" "peter" "kat" 2 3]))
 (println "Anonymous "
          ((fn [x] (* x 3)) 9)
          )

;more weird anonymous function
 (println (#(* % 3) 8) )
 (println (#(str " " %1 " - " %2 " - "%3 " others "%&) "ABC" "Buntha" "Abhi" 1 2 "ook"))


;;chain of anonymous function and passing funct to another

 (defn mainFunction
  [param]
  #(+ % param))

 (def result (mainFunction 3))

 (println "Call funct to another " (result 7))

 (def test_list
  ["Pongo" "Perdita" "Puppy 1" "Puppy 2"])

 (println (take 2 test_list))
 (println (let [variables (take 3 test_list)]
           variables))

 (println "Rest param " (let [[first & others] test_list] [first others]))

(def test_list2
 ["Abh" "buntha" "ABC" "Abhishek" "Bapun"])

;; let is an amazing way to name values
 (println "Associate" (let [[first & others] test_list
                test_list2 (conj test_list2 first)] [test_list2]) )


;;loop and recursion
(loop [iteration 0]
 (println (str "Iteration is " iteration))
 (if (> iteration 3)
  (println "Awesome its Done!")
  (recur (inc iteration)))
 )

 (println "Reducing with initial value" (reduce + 10 [1 2 3 4 5]))
(println (conj '(3 4) 2 1))
 (println (set '(:a :a :b :c :c :c :c :d :d)))

 (println "Set" (conj #{1 4 3} 2))

;;partial - another way to create function
 (println "Partial function " ( (partial + 5) 3))

 (println (#(str "Hello, " % "!") "Dave"))
 (println (#(second (reverse %) ) '(1 2 3 4)))


(defn testnth
 [values n]
 (loop [test n tmp-list values]
  (if (== test 0)
   (first tmp-list)
   (recur (dec test) (rest tmp-list))
   )
  ) )

;problem 21
 (println (
           (fn
            [values n]
            (loop [test n tmp-list values]
             (if (== test 0)
              (first tmp-list)
              (recur (dec test) (rest tmp-list))
              )
             ) )
           [0 1 2 3 4] 2) )

;; #22 Count a Sequence
(println "REDUCE " (reduce ( fn [x _] (inc x) ) 0  [0 1 2 3 4]
                           ) )
;;reverse a sequence #23
(println (reduce conj () [1 2 3 4]))

(defn reverse-sequence [params]
  (if (empty? params)
   []
   (conj (reverse-sequence (rest params)) (first params))
   )
 )

 (println "reverse a list with loop " (reverse-sequence '(1 2 3 4 5)))

;;sum it all up #24
(println "Sum of all number" (reduce + [1 2 3 4]))

;;find odd numbers #25
(println "find number" (filter odd? [1 2 3 4]))
(println "Find odd other " (filter #(== (mod % 2) 1) [1 2 3 4 5 6]))

(println " UR GET " (
                     (fn [n]
                      (take n (map second (iterate (fn [[a b]] [b (+ a b)]) [0 1] )))
                       )
                     8 ) )

;; fibonannic #26
(println "fibonannic #26n "  (fn [n] (take n (map second (iterate (fn [[a b]] [b (+ a b)]) [0 1] ))) ) 8 )

;; Palindrome Detector #27
(println "read-string " (seq (char-array "Abhishek") ))     ;very unfavourable method
(println "Convert to Sequence " (reverse (seq  [:foo :bar :foo :loo])) )
(println (apply str [1 2 3 45]))



(println "Palindrome Detector " (fn [n]
                                 (let [b (seq n)]
                                  (= b (reverse b)))) "racecar")

(println "map-cat "(mapcat reverse [[3 2 1 0] [6 5 4] [9 8 7]]))


(defn flatten1 [coll]
 (lazy-seq
  (when-let [s  (seq coll)]
   (if (coll? (first s))
    (concat (flatten1 (first s)) (flatten1 (rest s)))
    (cons (first s) (flatten1 (rest s)))
    ))
  ))
(println "Convert#28 Flatten Sequence " (flatten1 '((1 2) 3 [4 [5 6]]) ) )

(println "Flatten Other way" ((fn flatten* [x]
            (if (coll? x)
             (mapcat flatten* x)
             [x])) '((1 2) 3 [4 [5 6]])
           )  )


;;Get the Caps #29

(println "TEST " (filter #(Character/isUpperCase %) `(\$ \# \A \( \* \& \9 \8 \7 \Z \f)))

(println "Find UpperCase character " (fn [s] (reduce str (filter #(Character/isUpperCase %) s)))
                                       "abhISe, WorDliOp")




;;Compress the sequence #30
(println "JOIN=>" (str "Abhi" (subs "Chou" 1)))

(println "LOOPING "
   (fn callme [n]
     (if(empty? n)
      n
      (if (= (first n) (second n))
       (callme (rest n))
       (cons (first n) (callme (rest n)))
       )
      )
     ) "AbhiIIIIShekKKeiPpOPP"
   )

(println "map partiotion-by identity "  #(map first (partition-by identity %)) "AbhiIIIIShekKKeiPpOPP")



;;Pack a Sequence #31

(defn pack
       [n values]
       (if (empty? n)
        n
        (if (= n (first value))
         (pack  (rest n) (conj values n) )
         (cons n (pack (rest n) [])  )
         )
        )
 )

(println "Pack a Sequence " (flatten [[1 2] [1 2] [3 4]] ))
(println "Pack1 a Sequence " #(partition-by identity %)  [[1 2] [1 2] [3 4]])



;;https://www.4clojure.com/problem/solutions/31

;;Duplicate a Sequence
#(interleave % %)
(defn duplicate
  [n]
  (cons (repeat 1 (first n)) (rest duplicate))
  )

(println "repeat sequence " duplicate '(2 3))


;;Clojure 33 Replicate a Sequence
(println "repeat a seq " (#(repeat %2 %1) [2 3] 2) )
(println "mapcat "(#(mapcat (partial repeat %2) %) [[1 2] [3 4]] 2))

;Implement range #34

(println "Implement Range "(#(take (- %2 %1) (iterate inc %1) ) 5 8))

(println "Regex "(apply str (re-seq #"[A-Z]+" "bA1B3Ce")))

(println "find max "(
                      ( fn
                       [& args]
                       (reduce #(if (> %1 %2) %1 %2) args)
                       ) 6 2 5 8 1 ))

(println "Interleave seq 1"
         ((fn intleave [a b]
            (if (or (empty? a) (empty? b))
              []
              (concat [(first a) (first b)]
                       (intleave (rest a) (rest b)))
                )
              )
            [1 2 3] [:a :b :c])
           )


(println "Interleave seq 2"
         (#(flatten(map list %1 %2))[1 2] [3 4 5 6])
         )


(println "interpoase "(fn [sep coll]
                        (drop-last
                          (mapcat #(conj [] % sep) coll))) [1 2 3])

;;
(println "keep-indexed " (fn [col n] (keep-indexed (fn [i item] (when (not= (mod (inc i) n) 0) item)) col) [1 2 3 4 5 6 7 8] 3)
         )


(println "factorial "
         (fn [n]
           (loop [val n start 1]
             (if (zero? val)
               start
               (recur (dec val) (* start val))
               )
             )
           ) 5
         )

(println "factorial using reduce " #(reduce * (range 1 (inc %))) 5 )

(println "Reverse Interleave problem 43"  (#(apply map vector (partition %2 %1)) [1 2 3 4 5 6] 3) )

(println "Rotate Sequence problem 44 " (#(let [
                     [l r] (split-at (mod % (count %2)) %2)]
                (concat r l)) 2 [1 2 3 4 5] ))


(println "Iterate " (take 5 (iterate #(+ 3 %) 1))  )

(println "Flipping " ( #(fn [a b] (% b a)) 4))

(println "contains " (contains? [1 1 1 1 1] 4)) ;its about key not key

(println "some " (some #{2 7 6} [5 6 7 8]))

(println "Split a sequence 49. " (#(vector (vec (take %1 %2))
                                           (vec (drop %1 %2)))
                                   3 [1 2 3 4 5 6]))

(println "Split by Type # 50" (#(group-by type % )  [[1 2] :a [3 4] 5 6 :b]) )
(println "Split by Type # 50" (#(map second (group-by type % ))  [[1 2] :a [3 4] 5 6 :b]) ) ;map made the conj to a single sequence

(println " use of Let " (let [[a b & c  :as val] [1 2 3 4 5]] [a b c val]))


( println "custom distinct" (fn [s]
     ( reduce (fn [v n] (if (nil? (some #{n} v))
                          (conj v n)
                          v )) [] s)
     )
  [1 2 1 3 1 2 4] )


(println "Simple Recursion "
         (
           (fn foo [x]
               (when (> x 0)
                     (conj (foo (dec x)) x))) 5))


(println "Mod " (mod 2 8))

;;problem 61
#(into {} (map vector %1 %2 ))

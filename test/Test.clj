(ns Test)
(def headerLinks
  [{:name "Link1" :href "http://www.google.com" :class "glyphicon glyphicon-chevron-right"}
   {:name "Link2" :href "http://www.google.com" :class "glyphicon glyphicon-user"}
   {:name "Link3" :href "http://www.google.com" :class "glyphicon glyphicon-lock"}
   {:name "Link4" :href "http://www.google.com" :class "glyphicon glyphicon-cog"}]
  )



(defn newset []
  (doseq [record headerLinks
          [k v] record]
    (fn [k v]
      [:div
       [:val {:key k :value v}]
       ]
      ))
  )

(println newset)

(doseq [record headerLinks
        [k v] record]
  (

    ; println (str k " " v))

    )
  )


(defn dollme [n]
  [:div {:class n}
    [:h1 "Don"]
    ]
  )

(defn callme [n x name]
  [:div {:class n :id name}
   [:h1 x]
   ]
  )



(println (map #(callme (:class %1)(:href %1)(:name %1)) headerLinks))

(println "Partition " (partition 2 (range 9)))

(println "Count "  (#(apply map vector (partition %2 %1)) [1 2 3 4 5 6] 3) )

(println "Split a sequence 49. " (#(vector (vec (take %1 %2))
                            (vec (drop %1 %2))
                            )
                    3 [1 2 3 4 5 6]) )

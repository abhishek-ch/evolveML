 (ns testdev.register)

;http://yogthos.github.io/reagent-forms-example.html

(defn friend-source [text]
 (filter
  #(-> % (.toLowerCase %) (.indexOf text) (> -1))
  ["Alice" "Alan" "Bob" "Beth" "Jim" "Jane" "Kim" "Rob" "Zoe"]))

(defn test []
      [:div {:class "col-sm-8 col-sm-offset-2 text-center"}
       [:h2 "Debdoot Agency Group"]
       [:hr]
       [:h4 "Distributor & Wholeseller"]
       [:p
        "Connect or Contact us for knowing about our clients and Nature of Business"
        ]
       ])


(defn checkbox []
      [:div.row
       [:div.col-md-2 "does data binding make you happy?"]
       [:div.col-md-5
        [:input.form-control {:field :checkbox :id :happy-bindings :checked true}]]]
      )

(def months
  ["January" "February" "March" "April" "May" "June"
   "July" "August" "September" "October" "November" "December"])

;https://github.com/reagent-project/reagent-forms
(defn updatedlist []
    [:div
      [:select {:field :list :id :dob.day}
       (for [i (range 1 32)]
            [:option
             {:key (keyword (str i))
              :visible? #(let [month (get-in % [:dob :month])]
                              (cond
                                (< i 29) true
                                (< i 31) (not= month :February)
                                (= i 31) (some #{month} [:January :March :May :July :July :October :December])
                                :else false))}
             i])]
      [:select {:field :list :id :dob.month}
       (for [month months]
            [:option {:key (keyword month)} month])]
      [:select {:field :list :id :dob.year}
       (for [i (range 1950 (inc (.getFullYear (js/Date.))))]
            [:option {:key (keyword (str i))} i])]
     ]
      )

(defn list-val []
      [:div
       [:select.form-control {:field :list :id :many-options}
        [:option {:key :foo} "foo"]
        [:option {:key :bar} "bar"]
        [:option {:key :baz} "baz"]]
       ]
      )

(defn test_register []
[:div {:field :typeahead
       :id :ta
       :input-placeholder "pick a friend"
       :data-source friend-source
       :input-class "form-control"
       :list-class "typeahead-list"
       :item-class "typeahead-item"
       :highlight-class "highlighted"}])
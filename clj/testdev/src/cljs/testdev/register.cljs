(ns testdev.register

  (:require [reagent.core :as reagent :refer [atom]]
            [reagent-forms.core :refer [bind-fields init-field value-of]])
  )

;http://yogthos.github.io/reagent-forms-example.html

(defn friend-source [text]
 (filter
  #(-> % (.toLowerCase %) (.indexOf text) (> -1))
  ["Alice" "Alan" "Bob" "Beth" "Jim" "Jane" "Kim" "Rob" "Zoe"]))

(defn test_1 []
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
     [:div {:field :datepicker :id :birthday :date-format "yyyy/mm/dd" :inline true}]
     ]

      )


 (defn container-list-value []
   [:div.form-group
    {:field :container
     :visible? #(:show-name? %)}
    [:input {:field :text :id :first-name}]
    [:input {:field :text :id :last-name}]]
   )


 (defn row [label input]
   [:div.row
    [:div.col-md-2 [:label label]]
    [:div.col-md-5 input]])

 (defn form-template []
   [:div
    (row "first name" [:input {:field :text :id :first-name}])
    (row "last name" [:input {:field :text :id :last-name}])
    (row "age" [:input {:field :numeric :id :age}])
    (row "email" [:input {:field :email :id :email}])
    (row "comments" [:textarea {:field :textarea :id :comments}])])




 (def form-template-again
   [:div
    (row "first name"
      [:input.form-control {:field :text :id :first-name}])
    (row "last name"
      [:input.form-control {:field :text :id :last-name}])
    (row "age"
      [:input.form-control {:field :numeric :id :age}])
    (row "email"
      [:input.form-control {:field :email :id :email}])
    (row "comments"
      [:textarea.form-control {:field :textarea :id :comments}])])

 (defn forms-example []
   (let [doc (atom {:first-name "John" :last-name "Doe" :age 35})]
     (fn []
       [:div
        [:div.page-header [:h1 "Reagent Form"]]
        [bind-fields form-template-again doc]
        [:label (str @doc)]])))


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
import * as React from "react"
export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className="text-area"
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

const SearchsubdirTitleTextarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className="searchsubdir-title-text-area"
        ref={ref}
        {...props}
      />
    )
  }
)
SearchsubdirTitleTextarea.displayName = "SearchsubdirTitleTextarea"

const SearchsubdirTextarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className="searchsubdir-text-area"
        ref={ref}
        {...props}
      />
    )
  }
)
SearchsubdirTextarea.displayName = "SearchsubdirTextarea"

export { Textarea, SearchsubdirTitleTextarea, SearchsubdirTextarea}

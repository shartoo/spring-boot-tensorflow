package com.deepblue.tf_obj_detect;

public class RectF {
	 public float left;
	    public float top;
	    public float right;
	    public float bottom;
	    
	 public RectF(float left, float top, float right, float bottom) {
	        this.left = left;
	        this.top = top;
	        this.right = right;
	        this.bottom = bottom;
	    }
	    /**
	     * Create a new rectangle, initialized with the values in the specified
	     * rectangle (which is left unmodified).
	     *
	     * @param r The rectangle whose coordinates are copied into the new
	     *          rectangle.
	     */
	    public RectF(RectF r) {
	        if (r == null) {
	            left = top = right = bottom = 0.0f;
	        } else {
	            left = r.left;
	            top = r.top;
	            right = r.right;
	            bottom = r.bottom;
	        }
	    }
	    @Override
	    public boolean equals(Object o) {
	        if (this == o) return true;
	        if (o == null || getClass() != o.getClass()) return false;
	        RectF r = (RectF) o;
	        return left == r.left && top == r.top && right == r.right && bottom == r.bottom;
	    }
	    @Override
	    public int hashCode() {
	        int result = (left != +0.0f ? Float.floatToIntBits(left) : 0);
	        result = 31 * result + (top != +0.0f ? Float.floatToIntBits(top) : 0);
	        result = 31 * result + (right != +0.0f ? Float.floatToIntBits(right) : 0);
	        result = 31 * result + (bottom != +0.0f ? Float.floatToIntBits(bottom) : 0);
	        return result;
	    }
	    public String toString() {
	        return "RectF(" + left + ", " + top + ", "
	                      + right + ", " + bottom + ")";
	    }

}
